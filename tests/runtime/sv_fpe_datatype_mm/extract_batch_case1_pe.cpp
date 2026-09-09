// 将 batch case 1 的一个真实 PE 点积变成 rtl_cmodel_compare 可读取的向量。
// 该工具用于区分“PE 在相同输入下出错”与“CUTE 在任务切换时送错输入”。
#include "gendata.h"

#include <cstdint>
#include <cstdio>
#include <string>

// Symbols are renamed so this extractor can include the generated batch input.
#define a case1_a
#define b case1_b
#define c case1_c
#define d case1_d
#define gloden_c case1_golden
#define a_scale case1_a_scale
#define b_scale case1_b_scale
#include "generated/batch_case_1.h"

static void write_reverse_bytes(FILE* out, const uint8_t* data, int count) {
    // $readmemh 按数值最高位在文本最左侧读取；A/B 的逻辑 element[0] 位于
    // C 数组最低地址，因此要反转单字节 element 顺序，与 prepare_vectors.py
    // 的 bytes_to_sv_hex(..., elem_bytes=1) 保持一致。
    for (int index = count - 1; index >= 0; --index) {
        std::fprintf(out, "%02x", data[index]);
    }
    std::fprintf(out, "\n");
}

static void write_scale(FILE* out, const int8_t* data, int count, bool reverse) {
    // Scale bus 的 lane 方向与 A/B payload 不同。reverse-scale 模式用于将
    // CUTE 实际 scale bus 顺序与单 PE testbench 的 $readmemh 顺序校准。
    for (int offset = 0; offset < count; ++offset) {
        const int index = reverse ? count - 1 - offset : offset;
        std::fprintf(out, "%02x", static_cast<uint8_t>(data[index]));
    }
    std::fprintf(out, "\n");
}

int main(int argc, char** argv) {
    const char* output_dir = argc >= 2 ? argv[1] : "generated/case1_pe_vectors";
    const bool reverse_scale = argc >= 3 && std::string(argv[2]) == "reverse-scale";
    // 这是 batch 256-bit 回归中首个失配的逻辑输出坐标 D[2][11]。
    const int row = 2;
    const int column = 11;
    const int fpe_bytes = 32;
    const int scale_per_chunk = 4;

    FILE* a_out = std::fopen((std::string(output_dir) + "/A.txt").c_str(), "w");
    FILE* b_out = std::fopen((std::string(output_dir) + "/B.txt").c_str(), "w");
    FILE* c_out = std::fopen((std::string(output_dir) + "/C.txt").c_str(), "w");
    FILE* as_out = std::fopen((std::string(output_dir) + "/A_scale.txt").c_str(), "w");
    FILE* bs_out = std::fopen((std::string(output_dir) + "/B_scale.txt").c_str(), "w");
    FILE* d_out = std::fopen((std::string(output_dir) + "/D.txt").c_str(), "w");
    if (!a_out || !b_out || !c_out || !as_out || !bs_out || !d_out) return 2;

    // NVFP4 K=128 在 256-bit FPE 中拆为两个 K=64 transaction。第二条的
    // CAdd 是第一条的 partial sum，必须由同一 C-model 顺序计算得到。
    int32_t accum = case1_c[row][column];
    for (int chunk = 0; chunk < 2; ++chunk) {
        const uint8_t* a_ptr = reinterpret_cast<const uint8_t*>(case1_a[row]) + chunk * fpe_bytes;
        const uint8_t* b_ptr = reinterpret_cast<const uint8_t*>(case1_b[column]) + chunk * fpe_bytes;
        const int8_t* as_ptr = case1_a_scale + row * SCALE_COUNT + chunk * scale_per_chunk;
        const int8_t* bs_ptr = case1_b_scale + column * SCALE_COUNT + chunk * scale_per_chunk;
        const int32_t c_add = accum;
        accum = mymac_scale(const_cast<uint8_t*>(a_ptr), const_cast<int8_t*>(as_ptr),
                            const_cast<uint8_t*>(b_ptr), const_cast<int8_t*>(bs_ptr),
                            &accum, fpe_bytes, CMODEL_DTYPE);
        write_reverse_bytes(a_out, a_ptr, fpe_bytes);
        write_reverse_bytes(b_out, b_ptr, fpe_bytes);
        std::fprintf(c_out, "%08x\n", static_cast<uint32_t>(c_add));
        write_scale(as_out, as_ptr, scale_per_chunk, reverse_scale);
        write_scale(bs_out, bs_ptr, scale_per_chunk, reverse_scale);
        std::fprintf(d_out, "%08x\n", static_cast<uint32_t>(accum));
    }
    std::fclose(a_out); std::fclose(b_out); std::fclose(c_out);
    std::fclose(as_out); std::fclose(bs_out); std::fclose(d_out);
    return 0;
}
