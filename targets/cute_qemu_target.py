"""SDK-owned QEMU adapter. Inputs are exported hardware facts and requirements."""
import json
import re
import subprocess


def configure(facts, requirement, policy, backend):
    hart=requirement['hart'];cpu=facts['cpus'].get(hart,facts['cpus'].get(str(hart)))
    if cpu is None:raise ValueError('execution hart is absent')
    if cpu['xlen']!=backend['xlen'] or len(facts['cpus'])>backend['max_harts'] or hart!=0:
        raise ValueError('QEMU adapter does not support this hart topology')
    missing=set(requirement['isa'].get('custom_contracts',[]))-set(backend['contracts'])
    if missing:raise ValueError('QEMU custom contracts missing: '+', '.join(sorted(missing)))
    def version(value):
        m=re.fullmatch(r'(\d+)(?:[p.](\d+))?',value)
        if not m:raise ValueError('invalid QEMU ISA version bound')
        return int(m[1]),int(m[2] or 0)
    for name,bounds in requirement['isa'].get('versions',{}).items():
        known=backend.get('extension_versions',{}).get(name)
        if known is None:raise ValueError('QEMU ISA version coverage unknown: '+name)
        actual=version(known)
        if ('min' in bounds and actual<version(bounds['min'])) or ('max' in bounds and actual>version(bounds['max'])):
            raise ValueError('QEMU ISA version mismatch: '+name)
    ram=[r for r in facts['regions'] if r['kind']=='ram']
    if len(ram)!=1:
        raise ValueError('Spike backend requires one enabled RAM region')
    memory=ram[0]
    if not backend.get('configurable_dram_base') and memory['physical_base']!=int(backend['dram_base'],0):
        raise ValueError('Spike backend does not support this RAM base')
    if memory['bytes']%1024:raise ValueError('RAM extent must be a whole KiB')
    extensions=requirement['isa'].get('effective_extensions',requirement['isa']['extensions'])
    for name in extensions:
        actual=cpu.get('extension_versions',{}).get(name)
        if actual is not None:
            supported=backend.get('extension_versions',{}).get(name)
            if supported is None or version(supported)!=tuple(actual):
                raise ValueError('QEMU cannot establish the target ISA revision: '+name)
    if policy['coverage']=='full-target':
        extensions=cpu['extensions']
        missing=set(cpu.get('contracts',[]))-set(backend['contracts'])
        if missing:raise ValueError('full-target custom contracts not covered: '+', '.join(sorted(missing)))
        for name,hw_version in cpu.get('extension_versions',{}).items():
            if hw_version is not None:
                known=backend.get('extension_versions',{}).get(name)
                if known is None or version(known)!=tuple(hw_version):raise ValueError('full-target ISA version not covered: '+name)
    properties={e:True for e in extensions if len(e)==1 and e not in ('g','b')}
    for extension in extensions:
        if extension.startswith(('z','s','x')) and not extension.startswith('zvl'):
            properties[extension]=True
    if 'v' in cpu['extensions']:
        if cpu.get('vlen_bits') is None or cpu.get('elen_bits') is None:
            raise ValueError('exact VLEN/ELEN export required for QEMU')
        properties.update(v=True,vlen=cpu['vlen_bits'],elen=cpu['elen_bits'])
    properties['x-cute']=bool(requirement['isa'].get('custom_contracts'))
    if 'v' in requirement['isa'].get('versions',{}):properties['vext_spec']='v'+backend['extension_versions']['v']
    machine=backend['machine']
    if backend.get('configurable_dram_base'):
        machine+=f",cute-dram-base={memory['physical_base']}"
    clints=[d for d in facts.get('devices',[]) if set(d['compatible'])&set(backend['device_compatibles'])]
    if len(clints)>1:raise ValueError('Spike backend supports one CLINT')
    clint_base=int(backend.get('clint_base','0x02000000'),0)
    if clints and clints[0]['bytes']==backend.get('clint_bytes',65536):
        if backend.get('configurable_clint_base'):clint_base=clints[0]['base']
    if backend.get('configurable_clint_base'):
        machine+=f",cute-clint-base={clint_base}"
    private=[r for r in facts['regions'] if r['kind']=='tcm' and r['owner_hart']==hart]
    tcm=None
    if private:
        if len(private)!=1:raise ValueError('Spike adapter supports one private ordinary TCM')
        region=private[0]
        views=[v for v in region['views'] if v['client']=='rvv' and v['offset']==0 and v['bytes']==region['bytes']]
        if not views:raise ValueError('TCM needs a full RVV client window')
        view=min(views,key=lambda v:v['base'])
        if any(v['base']!=view['base'] for v in views):
            raise ValueError('Spike adapter cannot model additional TCM aliases')
        machine+=f",cute-tcm=true,cute-tcm-base={view['base']},cute-tcm-size={region['bytes']}"
        tcm={'region_id':region['id'],'client_base':view['base'],'bytes':region['bytes'],'backing_base':region['internal_base']}
    modeled=[d for d in clints if d['base']==clint_base and d['bytes']==backend.get('clint_bytes',65536)]
    ignored=[d for d in facts.get('devices',[]) if d not in modeled]
    for need in requirement.get('devices',[]):
        region=next((r for r in facts.get('configured_regions',[]) if r['id']==need['id']),None)
        if region is None or not any(d['path']==region['path'] and d['base']==region['base'] and d['bytes']==region['bytes'] for d in modeled):
            raise ValueError('QEMU does not model required address region: '+need['id'])
    if policy['coverage']=='full-target' and ignored:
        raise ValueError('QEMU full-target coverage missing devices: '+', '.join(d['path'] for d in ignored))
    cpu_arg=backend['cpu']+''.join(','+k+'='+('true' if v is True else 'false' if v is False else str(v)) for k,v in sorted(properties.items()))
    return {'machine':machine,'cpu_model':backend['cpu'],'cpu_properties':properties,'cpu':cpu_arg,
            'memory':str(memory['bytes']//1024)+'K','harts':len(facts['cpus']),'tcm':tcm,
            'coverage':{'mode':policy['coverage'],'ram_bytes':memory['bytes'],
                        'modeled_devices':modeled,'unmodeled_devices':ignored,
                        'allocation_reservations':facts.get('reservation_nodes',[]),
                        'contracts':requirement['isa'].get('custom_contracts',[])}}


def command(qemu, target, elf=None, checkpoints=None):
    cpu=target['cpu']
    if checkpoints:cpu+=',x-cute-checkpoints='+str(checkpoints)
    argv=[str(qemu),'-M',target['machine'],'-cpu',cpu,'-m',target['memory'],
          '-smp',str(target['harts']),'-nographic','-bios','none']
    if elf:argv+=['-kernel',str(elf)]
    return argv


def probe(qemu, target):
    queries=[{'execute':'qmp_capabilities','id':'caps'},
             {'execute':'query-cpu-model-expansion','id':'cpu','arguments':{'type':'full',
              'model':{'name':target['cpu_model'],'props':target['cpu_properties']}}},
             {'execute':'quit','id':'quit'}]
    argv=command(qemu,target)+['-S','-monitor','none','-serial','none','-qmp','stdio']
    run=subprocess.run(argv,input='\n'.join(json.dumps(q) for q in queries)+'\n',
                       capture_output=True,text=True,timeout=30)
    replies=[json.loads(line) for line in run.stdout.splitlines() if line.startswith('{')]
    errors=[r for r in replies if 'error' in r]
    cpu=next((r for r in replies if r.get('id')=='cpu'),None)
    if run.returncode or errors or not cpu or 'return' not in cpu:
        raise ValueError('QEMU target probe failed: '+run.stderr+' '+json.dumps(errors))
    actual=cpu['return']['model'].get('props',{})
    for name,value in target['cpu_properties'].items():
        if name in actual and actual[name]!=value:
            raise ValueError('QEMU did not honor property '+name)
    return {'verdict':'PASS','argv':argv,'cpu_expansion':cpu['return'],'stderr':run.stderr}
