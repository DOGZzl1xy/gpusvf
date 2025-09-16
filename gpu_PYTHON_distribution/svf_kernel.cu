// svf_kernel.cu
#define PI 3.1415926f

__device__ float annulus_weight(float altitude, float aziinterval) {
    float n = 90.0f;
    float steprad = (360.0f/aziinterval) * PI/180.0f;
    float annulus = 91.0f - altitude;
    float w = 1.0f/(2.0f*PI) * sinf(PI / (2.0f*n)) * sinf((PI * (2.0f * annulus - 1.0f)) / (2.0f * n));
    return steprad * w;
}

extern "C" __global__ void svf_shadowcasting_cupy(
    float* svf_Latt, float* dsm_Latt, float* tree_Latt, float scale, int imageW, int imageH, int rangeDist) // 新增 rangeDist 参数
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= imageW || y >= imageH) return;

    int index4 = x + y * imageW;

    // 如果当前位置有树且树高>0，认为站在树下，SVF=0 (性能优化，予以保留)
    if (tree_Latt[index4] > 0.0f) {
        svf_Latt[index4] = 0.03f;
        return;
    }

    // 穹顶分区参数
    float iangle[8] = {6, 18, 30, 42, 54, 66, 78, 90};
    float aziinterval[8] = {30, 30, 24, 24, 18, 12, 6, 1};
    float annulino[9] = {0, 12, 24, 36, 48, 60, 72, 84, 90};

    float svf = 0.0f;
    int idx = 0;
    // 计算每个分区的SVF
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < (int)aziinterval[i]; j++) {
            float altitude = iangle[i];
            float azimuth = j * (360.0f/aziinterval[i]);
            float altitude_rad = PI * altitude / 180.0f;
            float theta;
            // 太阳方位角转换
            if (azimuth < 90.0f && azimuth > 0.0f) {
                theta = PI * (90.0f - azimuth) / 180.0f;
            } else {
                theta = PI * (450.0f - azimuth) / 180.0f;
            }

            // --- 核心算法修改 ---
            float f = dsm_Latt[index4]; // 视线高度初始化为当前点高程
            float h_orig = f; // 保存原始地面高度

            // 修改1: 使用动态的 rangeDist，并从 radius=1.0f 开始避免采样自身
            for (float radius = 1.0f; radius < rangeDist; radius += 1.0f) {
                float x_f = x + radius * cosf(theta);
                float y_f = y - radius * sinf(theta);

                // 检查浮点坐标是否越界，并为双线性插值留出1个像素的边界
                if (x_f < 0.0f || x_f >= imageW - 1.0f || y_f < 0.0f || y_f >= imageH - 1.0f) break;

                // 修改2: 使用双线性插值法，解决边缘突变问题
                int x1 = (int)floorf(x_f);
                int y1 = (int)floorf(y_f);

                float x_frac = x_f - x1;
                float y_frac = y_f - y1;

                // 读取周围4个像素的高度和树高
                float h11 = dsm_Latt[x1 + y1 * imageW];
                float t11 = tree_Latt[x1 + y1 * imageW];
                if (t11 > 0.0f) h11 += 6.0f;

                float h12 = dsm_Latt[x1 + (y1 + 1) * imageW];
                float t12 = tree_Latt[x1 + (y1 + 1) * imageW];
                if (t12 > 0.0f) h12 += 6.0f;

                float h21 = dsm_Latt[(x1 + 1) + y1 * imageW];
                float t21 = tree_Latt[(x1 + 1) + y1 * imageW];
                if (t21 > 0.0f) h21 += 6.0f;

                float h22 = dsm_Latt[(x1 + 1) + (y1 + 1) * imageW];
                float t22 = tree_Latt[(x1 + 1) + (y1 + 1) * imageW];
                if (t22 > 0.0f) h22 += 6.0f;

                // 对高度进行双线性插值
                float h_top = h11 * (1.0f - x_frac) + h21 * x_frac;
                float h_bottom = h12 * (1.0f - x_frac) + h22 * x_frac;
                float height1 = h_top * (1.0f - y_frac) + h_bottom * y_frac;

                // 寻找路径上的最大遮挡高度角
                float temp = height1 - radius * tanf(altitude_rad) / scale;
                if (f < temp) f = temp;

                // 修改3: 移除了有风险的 "首次命中即中断" 的 break 语句
            }

            float sh = 0.0f;
            if (f == h_orig) { // 如果最大视线高度没有超过起始点高度，则天空可见
                sh = 1.0f;
            }
            // --------------------

            for (int k = (int)annulino[i] + 1; k < (int)annulino[i + 1] + 1; k++) {
                float weight = annulus_weight((float)k, aziinterval[i]);
                weight *= sh;
                svf += weight;
            }
            idx++;
        }
    }
    svf_Latt[index4] = svf;
}
