#ifndef CUTE_MODEL_STAGES_H
#define CUTE_MODEL_STAGES_H
/* Explicit, trace-visible stage kernels for the cutetest model migration.
 * B is N x K. Dimensions and row starts are multiples of 64 bytes.
 * Caller owns checkpoints, storage and buffer versions. No hidden scratch. */
#include <assert.h>
#include "cute_runtime.h"
#include "cute_sequence.h"
#include "cute_elementwise.h"
#include "cute_model_math.h"

static inline void cute_model_gemm(const int8_t *a, const int8_t *b,
    const int32_t *bias, int32_t *out, int m, int n, int k,
    int as, int bs, int os)
{
    assert(m>0 && n>0 && k>0 && m%64==0 && n%64==0 && k%64==0);
    assert(as>=k && bs>=k && os>=n && as%64==0 && bs%64==0 && os%16==0);
    for (int i=0;i<m;i+=64) for (int j=0;j<n;j+=64) {
        uint64_t task=cute_matmul(a+i*as,as,b+j*bs,bs,
            bias ? (void *)(bias+j) : out, bias ? n*4 : os*4,
            out+i*os+j,os*4,64,64,k,0,bias ? 2 : 1,0,0);
        cute_wait_task(task);
    }
}
static inline vint8m1_t cute_model_pack(vint32m4_t x,size_t vl)
{
    x=__riscv_vmax_vx_i32m4(__riscv_vmin_vx_i32m4(x,127,vl),-128,vl);
    return __riscv_vncvt_x_x_w_i8m1(__riscv_vncvt_x_x_w_i16m2(x,vl),vl);
}
static inline void cute_model_i8(const int32_t *x,int8_t *y,int count,int gelu)
{
    for(int c=0;c<count;){size_t vl=__riscv_vsetvl_e32m4(count-c);
        vint32m4_t q=__riscv_vle32_v_i32m4(x+c,vl);
        if(gelu){ /* cutetest I-GELU with scale=1: q1=qc=-6,qb=-2 */
            vint32m4_t sign=__riscv_vmerge_vxm_i32m4(__riscv_vmv_v_x_i32m4(1,vl),-1,__riscv_vmslt_vx_i32m4_b8(q,0,vl),vl);
            vint32m4_t t=__riscv_vmul_vv_i32m4(q,sign,vl);
            t=__riscv_vadd_vx_i32m4(__riscv_vmin_vx_i32m4(t,2,vl),-2,vl);
            t=__riscv_vadd_vx_i32m4(__riscv_vmul_vv_i32m4(t,t,vl),-6,vl);
            t=__riscv_vadd_vx_i32m4(__riscv_vmul_vv_i32m4(t,sign,vl),-6,vl);
            q=__riscv_vmul_vv_i32m4(q,t,vl);
        }
        __riscv_vse8_v_i8m1(y+c,cute_model_pack(q,vl),vl);c+=vl;
    }
}
static inline int32_t cute_model_isqrt(int32_t n)
{
    if(n<=0)return 0;
    int bits=0; for(int32_t v=n;v;v/=2)++bits;
    int32_t x=1<<((bits+1)/2);
    for(;;){int32_t next=(x+n/x)/2;if(next>=x)return x;x=next;}
}
static inline void cute_model_layernorm(const int32_t *x,const int8_t *res,int8_t *y,int rows,int cols)
{
    for(int r=0;r<rows;r++){
        int32_t sum=0,ss=0;
        for(int c=0;c<cols;){size_t vl=__riscv_vsetvl_e32m4(cols-c);
            vint32m4_t v=__riscv_vle32_v_i32m4(x+r*cols+c,vl);
            sum=(int32_t)((uint32_t)sum+(uint32_t)__riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32m4_i32m1(v,__riscv_vmv_v_x_i32m1(0,1),vl)));c+=vl;}
        int32_t mean=sum/cols;
        for(int c=0;c<cols;){size_t vl=__riscv_vsetvl_e32m4(cols-c);
            vint32m4_t v=__riscv_vsub_vx_i32m4(__riscv_vle32_v_i32m4(x+r*cols+c,vl),mean,vl);
            v=__riscv_vmul_vv_i32m4(v,v,vl);
            ss=(int32_t)((uint32_t)ss+(uint32_t)__riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32m4_i32m1(v,__riscv_vmv_v_x_i32m1(0,1),vl)));c+=vl;}
        int32_t sd=cute_model_isqrt(ss/cols); if(!sd)sd=1; float factor=1.0f/(float)sd;
        for(int c=0;c<cols;){size_t vl=__riscv_vsetvl_e32m4(cols-c);
            vint32m4_t v=__riscv_vsub_vx_i32m4(__riscv_vle32_v_i32m4(x+r*cols+c,vl),mean,vl);
            v=__riscv_vfcvt_x_f_v_i32m4(__riscv_vfmul_vf_f32m4(__riscv_vfcvt_f_x_v_f32m4(v,vl),factor,vl),vl);
            vint8m1_t narrow=cute_model_pack(v,vl);
            vint16m2_t add=__riscv_vwadd_vv_i16m2(narrow,__riscv_vle8_v_i8m1(res+r*cols+c,vl),vl);
            add=__riscv_vmax_vx_i16m2(__riscv_vmin_vx_i16m2(add,127,vl),-128,vl);
            __riscv_vse8_v_i8m1(y+r*cols+c,__riscv_vncvt_x_x_w_i8m1(add,vl),vl);c+=vl;}
    }
}
/* Integer softmax scale=.5, exp scratch is an explicit task output. */
static inline void cute_model_softmax(const int32_t *x,int32_t *ex,int8_t *y,int rows,int cols)
{
    for(int r=0;r<rows;r++){
        int32_t maximum=INT32_MIN,sum=0;
        for(int c=0;c<cols;){size_t vl=__riscv_vsetvl_e32m4(cols-c);
            maximum=__riscv_vmv_x_s_i32m1_i32(__riscv_vredmax_vs_i32m4_i32m1(__riscv_vle32_v_i32m4(x+r*cols+c,vl),__riscv_vmv_v_x_i32m1(maximum,1),vl));c+=vl;}
        for(int c=0;c<cols;){size_t vl=__riscv_vsetvl_e32m4(cols-c);
            vint32m4_t q=__riscv_vsub_vx_i32m4(__riscv_vle32_v_i32m4(x+r*cols+c,vl),maximum,vl);
            vint32m4_t z=__riscv_vsra_vx_i32m4(__riscv_vmul_vx_i32m4(__riscv_vrsub_vx_i32m4(q,0,vl),65536,vl),16,vl);
            q=__riscv_vadd_vx_i32m4(__riscv_vadd_vv_i32m4(q,z,vl),2,vl);
            q=__riscv_vadd_vx_i32m4(__riscv_vmul_vv_i32m4(q,q,vl),3,vl);
            q=__riscv_vsra_vv_i32m4(q,__riscv_vreinterpret_v_i32m4_u32m4(z),vl);
            __riscv_vse32_v_i32m4(ex+r*cols+c,q,vl);
            sum=(int32_t)((uint32_t)sum+(uint32_t)__riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32m4_i32m1(q,__riscv_vmv_v_x_i32m1(0,1),vl)));c+=vl;}
        __asm__ volatile("fence rw,rw":::"memory");
        float factor=127.0f/(float)sum;
        for(int c=0;c<cols;){size_t vl=__riscv_vsetvl_e32m4(cols-c);
            vfloat32m4_t v=__riscv_vfcvt_f_x_v_f32m4(__riscv_vle32_v_i32m4(ex+r*cols+c,vl),vl);
            __riscv_vse8_v_i8m1(y+r*cols+c,cute_model_pack(__riscv_vfcvt_x_f_v_i32m4(__riscv_vfmul_vf_f32m4(v,factor,vl),vl),vl),vl);c+=vl;}
    }
}
/* Ordered reduction gives one documented FP32 sum order across QEMU and RTL. */
static inline void cute_model_rms(const float *x,const float *weight,float *y,int rows,int cols)
{
    for(int r=0;r<rows;r++){
        /* Fixed 64-lane groups, then an ordered reduction of those lanes.
         * All migration shapes have cols divisible by 64 and VLEN=512. */
        size_t reduction_vl=__riscv_vsetvl_e32m4(64);
        vfloat32m4_t partial=__riscv_vfmv_v_f_f32m4(0.0f,reduction_vl);
        for(int c=0;c<cols;c+=64){
            vfloat32m4_t v=__riscv_vle32_v_f32m4(x+r*cols+c,reduction_vl);
            partial=__riscv_vfadd_vv_f32m4(partial,__riscv_vfmul_vv_f32m4(v,v,reduction_vl),reduction_vl);
        }
        float sum=__riscv_vfmv_f_s_f32m1_f32(__riscv_vfredosum_vs_f32m4_f32m1(partial,__riscv_vfmv_v_f_f32m1(0.0f,1),reduction_vl));
        float inv=1.0f/cute_fast_sqrt(sum/(float)cols+1.0e-5f);
        for(int c=0;c<cols;){size_t vl=__riscv_vsetvl_e32m4(cols-c);
            vfloat32m4_t v=__riscv_vfmul_vf_f32m4(__riscv_vle32_v_f32m4(x+r*cols+c,vl),inv,vl);
            v=__riscv_vfmul_vv_f32m4(v,__riscv_vle32_v_f32m4(weight+c,vl),vl);
            __riscv_vse32_v_f32m4(y+r*cols+c,v,vl);c+=vl;}
    }
}
static inline void cute_model_quant(const float *x,float *scale,int8_t *y,int rows,int cols)
{
    for(int r=0;r<rows;r++){
        float maximum=0;
        for(int c=0;c<cols;){size_t vl=__riscv_vsetvl_e32m4(cols-c);
            maximum=__riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m4_f32m1(__riscv_vfabs_v_f32m4(__riscv_vle32_v_f32m4(x+r*cols+c,vl),vl),__riscv_vfmv_v_f_f32m1(maximum,1),vl));c+=vl;}
        float s=maximum/127.0f,inv=s ? 1.0f/s : 0;
        __riscv_vse32_v_f32m1(scale+r,__riscv_vfmv_v_f_f32m1(s,1),1);
        for(int c=0;c<cols;){size_t vl=__riscv_vsetvl_e32m4(cols-c);
            vfloat32m4_t v=__riscv_vfmul_vf_f32m4(__riscv_vle32_v_f32m4(x+r*cols+c,vl),inv,vl);
            __riscv_vse8_v_i8m1(y+r*cols+c,cute_model_pack(__riscv_vfcvt_x_f_v_i32m4(v,vl),vl),vl);c+=vl;}
    }
}
static inline void cute_model_dequant(const int32_t *x,const float *scales,float ws,float *y,int rows,int cols)
{
    for(int r=0;r<rows;r++)for(int c=0;c<cols;){size_t vl=__riscv_vsetvl_e32m4(cols-c);
        vfloat32m4_t v=__riscv_vfcvt_f_x_v_f32m4(__riscv_vle32_v_i32m4(x+r*cols+c,vl),vl);
        v=__riscv_vfmul_vf_f32m4(v,scales[r]*ws,vl);
        __riscv_vse32_v_f32m4(y+r*cols+c,v,vl);c+=vl;}
}
static inline void cute_model_transpose_i8(const int8_t *x,int8_t *y,int rows,int cols)
{
    for(int c=0;c<cols;c++)for(int r=0;r<rows;){size_t vl=__riscv_vsetvl_e8m1(rows-r);
        __riscv_vse8_v_i8m1(y+c*rows+r,__riscv_vlse8_v_i8m1(x+r*cols+c,cols,vl),vl);r+=vl;}
}
static inline void cute_model_transpose_f16(const float *x,_Float16 *y,int rows,int cols)
{
    for(int c=0;c<cols;c++)for(int r=0;r<rows;){size_t vl=__riscv_vsetvl_e32m4(rows-r);
        __riscv_vse16_v_f16m2(y+c*rows+r,__riscv_vfncvt_f_f_w_f16m2(__riscv_vlse32_v_f32m4(x+r*cols+c,cols*4,vl),vl),vl);r+=vl;}
}
static inline void cute_model_rope(const float *x,_Float16 *y,const float *theta,int rows,int cols,int head)
{
    for(int r=0;r<rows;r++)for(int h=0;h<cols;h+=head)for(int k=0;k<head/2;){
        size_t vl=__riscv_vsetvl_e32m1(head/2-k);
        vfloat32m1_t angle=__riscv_vfmul_vf_f32m1(__riscv_vle32_v_f32m1(theta+k,vl),(float)r,vl);
        vfloat32m1_t si=cute_model_vec_sin(angle,vl),co=cute_model_vec_cos(angle,vl);
        vfloat32m1_t re=__riscv_vlse32_v_f32m1(x+r*cols+h+2*k,8,vl);
        vfloat32m1_t im=__riscv_vlse32_v_f32m1(x+r*cols+h+2*k+1,8,vl);
        vfloat32m1_t real=__riscv_vfmsub_vv_f32m1(re,co,__riscv_vfmul_vv_f32m1(im,si,vl),vl);
        vfloat32m1_t imag=__riscv_vfmacc_vv_f32m1(__riscv_vfmul_vv_f32m1(re,si,vl),im,co,vl);
        __riscv_vse16_v_f16mf2(y+r*cols+h+k,__riscv_vfncvt_f_f_w_f16mf2(real,vl),vl);
        __riscv_vse16_v_f16mf2(y+r*cols+h+head/2+k,__riscv_vfncvt_f_f_w_f16mf2(imag,vl),vl);k+=vl;
    }
}
static inline void cute_model_silu(const float *x,float *y,int count)
{
    for(int i=0;i<count;){size_t vl=__riscv_vsetvl_e32m1(count-i);
        vfloat32m1_t v=__riscv_vle32_v_f32m1(x+i,vl);
        vfloat32m1_t den=__riscv_vfadd_vf_f32m1(cute_model_vec_exp(__riscv_vfneg_v_f32m1(v,vl),vl),1.0f,vl);
        __riscv_vse32_v_f32m1(y+i,__riscv_vfmul_vv_f32m1(v,cute_model_vec_recip_approx(den,vl),vl),vl);i+=vl;
    }
}
#endif
