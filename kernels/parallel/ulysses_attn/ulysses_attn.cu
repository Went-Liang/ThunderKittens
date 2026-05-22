/*
    Note that self-attention computation uses FlashAttention, since no overlapping is needed. 
    See the Python script for more details.
*/

#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

using namespace kittens;

namespace all_to_all {

static constexpr int ROW_BLOCK_SIZE = 16; // support 128 head count

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int MIN_BLOCKS_PER_SM = 8;
    static constexpr int NUM_THREADS = 1;
};

template <int NUM_DEVICES, int COL_BLOCK_SIZE_>
struct globals {
    static constexpr int ROW_BLOCK_SIZE = all_to_all::ROW_BLOCK_SIZE;
    static constexpr int COL_BLOCK_SIZE = COL_BLOCK_SIZE_;

    using shared_tile = st_bf<ROW_BLOCK_SIZE, COL_BLOCK_SIZE>;
    using parallel_layout = pgl<gl<bf16, -1, -1, -1, -1, shared_tile>, NUM_DEVICES, false>;

    parallel_layout output;
    parallel_layout input;
    const int dev_idx;

    __host__ inline dim3 grid() const {
        return dim3((input.cols() / globals::COL_BLOCK_SIZE) *
                    (input.rows() / globals::ROW_BLOCK_SIZE) *
                    input.depth() * input.batch());
    }

    __host__ inline int dynamic_shared_memory() const {
        return static_cast<int>(sizeof(shared_tile) + 1024);
    }
};

template <int SCATTER_AXIS, int GATHER_AXIS, int NUM_DEVICES, int COL_BLOCK_SIZE>
__device__ inline void kernel(const globals<NUM_DEVICES, COL_BLOCK_SIZE> &G) {
    static_assert(0 <= SCATTER_AXIS && SCATTER_AXIS < 4 && 0 <= GATHER_AXIS && GATHER_AXIS < 4, 
        "Scatter and gather axes must be 0, 1, 2, or 3");
    static_assert(SCATTER_AXIS != GATHER_AXIS, "Scatter and gather axes must be different");

    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int*)&__shm[0]);
    typename globals<NUM_DEVICES, COL_BLOCK_SIZE>::shared_tile &tile =
        allocator.allocate<typename globals<NUM_DEVICES, COL_BLOCK_SIZE>::shared_tile>();

    // Calculate the input indices
    int task_idx = blockIdx.x;
    int batch_idx = task_idx / (G.input.depth() * (G.input.rows() / globals<NUM_DEVICES, COL_BLOCK_SIZE>::ROW_BLOCK_SIZE) * (G.input.cols() / globals<NUM_DEVICES, COL_BLOCK_SIZE>::COL_BLOCK_SIZE));
    task_idx %= (G.input.depth() * (G.input.rows() / globals<NUM_DEVICES, COL_BLOCK_SIZE>::ROW_BLOCK_SIZE) * (G.input.cols() / globals<NUM_DEVICES, COL_BLOCK_SIZE>::COL_BLOCK_SIZE));
    int depth_idx = task_idx / (G.input.rows() / globals<NUM_DEVICES, COL_BLOCK_SIZE>::ROW_BLOCK_SIZE * (G.input.cols() / globals<NUM_DEVICES, COL_BLOCK_SIZE>::COL_BLOCK_SIZE));
    task_idx %= (G.input.rows() / globals<NUM_DEVICES, COL_BLOCK_SIZE>::ROW_BLOCK_SIZE * (G.input.cols() / globals<NUM_DEVICES, COL_BLOCK_SIZE>::COL_BLOCK_SIZE));
    int row_block_idx = task_idx / (G.input.cols() / globals<NUM_DEVICES, COL_BLOCK_SIZE>::COL_BLOCK_SIZE);
    task_idx %= (G.input.cols() / globals<NUM_DEVICES, COL_BLOCK_SIZE>::COL_BLOCK_SIZE);
    int col_block_idx = task_idx;

    if constexpr (SCATTER_AXIS == 2) {
        // Interleave row-scatter traffic across destination GPUs instead of draining one GPU at a time.
        const int output_row_blocks = G.output.rows() / globals<NUM_DEVICES, COL_BLOCK_SIZE>::ROW_BLOCK_SIZE;
        const int interleaved_dst_dev_idx = row_block_idx % NUM_DEVICES;
        const int row_block_idx_in_dst = row_block_idx / NUM_DEVICES;
        row_block_idx = interleaved_dst_dev_idx * output_row_blocks + row_block_idx_in_dst;
    }

    // Load input data (assume a single-threaded block)
    __shared__ semaphore arrived;
    init_semaphore(arrived, 0, 1);
    tma::expect_bytes(arrived, sizeof(tile));
    tma::load_async(tile, G.input[G.dev_idx], {batch_idx, depth_idx, row_block_idx, col_block_idx}, arrived);

    // Calculate the output indices
    int dst_dev_idx;

    if constexpr (SCATTER_AXIS == 0) {
        dst_dev_idx = batch_idx / G.output.batch();
        batch_idx %= G.output.batch();
    } else if constexpr (SCATTER_AXIS == 1) {
        dst_dev_idx = depth_idx / G.output.depth();
        depth_idx %= G.output.depth();
    } else if constexpr (SCATTER_AXIS == 2) {
        dst_dev_idx = row_block_idx / (G.output.rows() / globals<NUM_DEVICES, COL_BLOCK_SIZE>::ROW_BLOCK_SIZE);
        row_block_idx %= (G.output.rows() / globals<NUM_DEVICES, COL_BLOCK_SIZE>::ROW_BLOCK_SIZE);
    } else {
        dst_dev_idx = col_block_idx / (G.output.cols() / globals<NUM_DEVICES, COL_BLOCK_SIZE>::COL_BLOCK_SIZE);
        col_block_idx %= (G.output.cols() / globals<NUM_DEVICES, COL_BLOCK_SIZE>::COL_BLOCK_SIZE);
    }

    if constexpr (GATHER_AXIS == 0) {
        batch_idx += G.input.batch() * G.dev_idx;
    } else if constexpr (GATHER_AXIS == 1) {
        depth_idx += G.input.depth() * G.dev_idx;
    } else if constexpr (GATHER_AXIS == 2) {
        row_block_idx += (G.input.rows() / globals<NUM_DEVICES, COL_BLOCK_SIZE>::ROW_BLOCK_SIZE) * G.dev_idx;
    } else {
        col_block_idx += (G.input.cols() / globals<NUM_DEVICES, COL_BLOCK_SIZE>::COL_BLOCK_SIZE) * G.dev_idx;
    }

    // Wait for inputs to arrive and store data to destination device
    wait(arrived, 0);
    tma::store_async(G.output[dst_dev_idx], tile, 
        {batch_idx, depth_idx, row_block_idx, col_block_idx});
}

} // namespace all_to_all

namespace all_to_all_barrier {

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_BLOCKS = 1;
    static constexpr int NUM_THREADS = 1;
    static constexpr int DYNAMIC_SHARED_MEMORY = 0;
};

template <int NUM_DEVICES>
struct globals {
    barrier_t<NUM_DEVICES> barrier;
    const int dev_idx;
};

template <int NUM_DEVICES>
__device__ inline void kernel(const globals<NUM_DEVICES> &G) {
    barrier_all(G.barrier, {0}, G.dev_idx);
}

} // namespace all_to_all_barrier

template <int NUM_DEVICES, int COL_BLOCK_SIZE>
void dispatch_all_to_all(
    typename all_to_all::globals<NUM_DEVICES, COL_BLOCK_SIZE> &all_to_all_G,
    int scatter_axis,
    int gather_axis
) {
    if (scatter_axis == 0 && gather_axis == 1)
        kittens::py::launch_kernel<all_to_all::config, all_to_all::globals<NUM_DEVICES, COL_BLOCK_SIZE>, all_to_all::kernel<0, 1, NUM_DEVICES, COL_BLOCK_SIZE>>(all_to_all_G);
    else if (scatter_axis == 0 && gather_axis == 2)
        kittens::py::launch_kernel<all_to_all::config, all_to_all::globals<NUM_DEVICES, COL_BLOCK_SIZE>, all_to_all::kernel<0, 2, NUM_DEVICES, COL_BLOCK_SIZE>>(all_to_all_G);
    else if (scatter_axis == 0 && gather_axis == 3)
        kittens::py::launch_kernel<all_to_all::config, all_to_all::globals<NUM_DEVICES, COL_BLOCK_SIZE>, all_to_all::kernel<0, 3, NUM_DEVICES, COL_BLOCK_SIZE>>(all_to_all_G);
    else if (scatter_axis == 1 && gather_axis == 0)
        kittens::py::launch_kernel<all_to_all::config, all_to_all::globals<NUM_DEVICES, COL_BLOCK_SIZE>, all_to_all::kernel<1, 0, NUM_DEVICES, COL_BLOCK_SIZE>>(all_to_all_G);
    else if (scatter_axis == 1 && gather_axis == 2)
        kittens::py::launch_kernel<all_to_all::config, all_to_all::globals<NUM_DEVICES, COL_BLOCK_SIZE>, all_to_all::kernel<1, 2, NUM_DEVICES, COL_BLOCK_SIZE>>(all_to_all_G);
    else if (scatter_axis == 1 && gather_axis == 3)
        kittens::py::launch_kernel<all_to_all::config, all_to_all::globals<NUM_DEVICES, COL_BLOCK_SIZE>, all_to_all::kernel<1, 3, NUM_DEVICES, COL_BLOCK_SIZE>>(all_to_all_G);
    else if (scatter_axis == 2 && gather_axis == 0)
        kittens::py::launch_kernel<all_to_all::config, all_to_all::globals<NUM_DEVICES, COL_BLOCK_SIZE>, all_to_all::kernel<2, 0, NUM_DEVICES, COL_BLOCK_SIZE>>(all_to_all_G);
    else if (scatter_axis == 2 && gather_axis == 1)
        kittens::py::launch_kernel<all_to_all::config, all_to_all::globals<NUM_DEVICES, COL_BLOCK_SIZE>, all_to_all::kernel<2, 1, NUM_DEVICES, COL_BLOCK_SIZE>>(all_to_all_G);
    else if (scatter_axis == 2 && gather_axis == 3)
        kittens::py::launch_kernel<all_to_all::config, all_to_all::globals<NUM_DEVICES, COL_BLOCK_SIZE>, all_to_all::kernel<2, 3, NUM_DEVICES, COL_BLOCK_SIZE>>(all_to_all_G);
    else if (scatter_axis == 3 && gather_axis == 0)
        kittens::py::launch_kernel<all_to_all::config, all_to_all::globals<NUM_DEVICES, COL_BLOCK_SIZE>, all_to_all::kernel<3, 0, NUM_DEVICES, COL_BLOCK_SIZE>>(all_to_all_G);
    else if (scatter_axis == 3 && gather_axis == 1)
        kittens::py::launch_kernel<all_to_all::config, all_to_all::globals<NUM_DEVICES, COL_BLOCK_SIZE>, all_to_all::kernel<3, 1, NUM_DEVICES, COL_BLOCK_SIZE>>(all_to_all_G);
    else if (scatter_axis == 3 && gather_axis == 2)
        kittens::py::launch_kernel<all_to_all::config, all_to_all::globals<NUM_DEVICES, COL_BLOCK_SIZE>, all_to_all::kernel<3, 2, NUM_DEVICES, COL_BLOCK_SIZE>>(all_to_all_G);
    else
        TORCH_CHECK(false, "Invalid scatter and gather axes");
}

template <int NUM_DEVICES, int COL_BLOCK_SIZE>
void run_all_to_all(
    kittens::py::TKParallelTensor &output,
    kittens::py::TKParallelTensor &input,
    kittens::py::TKParallelTensor &barrier,
    int scatter_axis,
    int gather_axis
) {
    using globals = all_to_all::globals<NUM_DEVICES, COL_BLOCK_SIZE>;
    globals all_to_all_G {
        .output = kittens::py::parallel_tensor_to_pgl<typename globals::parallel_layout>(output),
        .input  = kittens::py::parallel_tensor_to_pgl<typename globals::parallel_layout>(input),
        .dev_idx = input.local_rank_
    };
    all_to_all_barrier::globals<NUM_DEVICES> barrier_G {
        .barrier = kittens::py::parallel_tensor_to_pgl<barrier_t<NUM_DEVICES>>(barrier),
        .dev_idx = barrier.local_rank_
    };

    kittens::py::launch_kernel<all_to_all_barrier::config, all_to_all_barrier::globals<NUM_DEVICES>, all_to_all_barrier::kernel<NUM_DEVICES>>(barrier_G);
    dispatch_all_to_all<NUM_DEVICES, COL_BLOCK_SIZE>(all_to_all_G, scatter_axis, gather_axis);
    kittens::py::launch_kernel<all_to_all_barrier::config, all_to_all_barrier::globals<NUM_DEVICES>, all_to_all_barrier::kernel<NUM_DEVICES>>(barrier_G);
}

int choose_col_block(
    const kittens::py::TKParallelTensor &output,
    const kittens::py::TKParallelTensor &input
) {
    TORCH_CHECK(input.data_.dim() > 0 && output.data_.dim() > 0,
                "Input and output must have at least one dimension");

    const int input_cols = input.data_.size(input.data_.dim() - 1);
    const int output_cols = output.data_.size(output.data_.dim() - 1);
    if (input_cols % 128 == 0 && output_cols % 128 == 0)
        return 128;
    if (input_cols % 64 == 0 && output_cols % 64 == 0)
        return 64;
    if (input_cols % 32 == 0 && output_cols % 32 == 0)
        return 32;

    TORCH_CHECK(false, "Input and output column dimensions must both be divisible by 32. Got input cols: ",
                input_cols, ", output cols: ", output_cols);
    return 0;
}

int64_t logical_4d_size(const at::Tensor &tensor, int axis) {
    const int tensor_axis = axis - (4 - tensor.dim());
    return tensor_axis < 0 ? 1 : tensor.size(tensor_axis);
}

void check_all_to_all_shapes(
    const kittens::py::TKParallelTensor &output,
    const kittens::py::TKParallelTensor &input,
    int num_devices,
    int scatter_axis,
    int gather_axis
) {
    const int dim = input.data_.dim();
    TORCH_CHECK(0 <= scatter_axis && scatter_axis < dim && 0 <= gather_axis && gather_axis < dim,
                "Scatter and gather axes must be valid for input dim");
    TORCH_CHECK(scatter_axis != gather_axis, "Scatter and gather axes must be different");
    TORCH_CHECK(input.data_.size(scatter_axis) % num_devices == 0,
                "Input scatter dimension must be divisible by num_devices");

    for (int axis = 0; axis < dim; ++axis) {
        int64_t expected = input.data_.size(axis);
        if (axis == scatter_axis)
            expected /= num_devices;
        if (axis == gather_axis)
            expected *= num_devices;
        TORCH_CHECK(output.data_.size(axis) == expected,
                    "Output shape mismatch at axis ", axis, ". Expected ", expected,
                    ", got ", output.data_.size(axis));
    }

    TORCH_CHECK(logical_4d_size(input.data_, 2) % all_to_all::ROW_BLOCK_SIZE == 0 &&
                logical_4d_size(output.data_, 2) % all_to_all::ROW_BLOCK_SIZE == 0,
                "Input and output row dimensions must both be divisible by 16");
}

using AllToAllLauncher = void (*)(
    kittens::py::TKParallelTensor &,
    kittens::py::TKParallelTensor &,
    kittens::py::TKParallelTensor &,
    int,
    int
);

struct DispatchEntry {
    int num_devices;
    int col_block_size;
    AllToAllLauncher launcher;
};

static constexpr DispatchEntry DISPATCH_TABLE[] = {
    {2, 128, &run_all_to_all<2, 128>},
    {2,  64, &run_all_to_all<2,  64>},
    {2,  32, &run_all_to_all<2,  32>},
    {4, 128, &run_all_to_all<4, 128>},
    {4,  64, &run_all_to_all<4,  64>},
    {4,  32, &run_all_to_all<4,  32>},
    {8, 128, &run_all_to_all<8, 128>},
    {8,  64, &run_all_to_all<8,  64>},
    {8,  32, &run_all_to_all<8,  32>},
};

void launch_all_to_all(
    int num_devices,
    int col_block_size,
    kittens::py::TKParallelTensor &output,
    kittens::py::TKParallelTensor &input,
    kittens::py::TKParallelTensor &barrier,
    int scatter_axis,
    int gather_axis
) {
    for (const DispatchEntry &entry : DISPATCH_TABLE) {
        if (entry.num_devices == num_devices && entry.col_block_size == col_block_size) {
            entry.launcher(output, input, barrier, scatter_axis, gather_axis);
            return;
        }
    }

    TORCH_CHECK(false, "Unsupported dispatch combination. num_devices: ", num_devices,
                ", col_block_size: ", col_block_size);
}

void entrypoint(
    kittens::py::TKParallelTensor &output,
    kittens::py::TKParallelTensor &input,
    kittens::py::TKParallelTensor &barrier,
    int scatter_axis,
    int gather_axis
) {
    TORCH_CHECK(output.data_.dim() == input.data_.dim(), 
        "Output and input must have the same number of dimensions");
    const int num_devices = input.local_world_size_;
    TORCH_CHECK(num_devices == output.local_world_size_,
                "Input and output must have the same number of devices");
    TORCH_CHECK(num_devices == barrier.local_world_size_,
                "Input and barrier must have the same number of devices");
    TORCH_CHECK(input.broker_id_ == output.broker_id_ && input.broker_id_ == barrier.broker_id_,
                "Input, output, and barrier must use the same TK broker namespace. Got input broker_id: ",
                input.broker_id_, ", output broker_id: ", output.broker_id_,
                ", barrier broker_id: ", barrier.broker_id_);
    check_all_to_all_shapes(output, input, num_devices, scatter_axis, gather_axis);

    // Adjust scatter_axis and gather_axis if dimension is less than 4
    int dim = input.data_.dim();
    if (dim < 4) {
        scatter_axis += (4 - dim);
        gather_axis += (4 - dim);
    }

    const int col_block_size = choose_col_block(output, input);
    launch_all_to_all(num_devices, col_block_size, output, input, barrier, scatter_axis, gather_axis);
}

#include <torch/csrc/utils/pybind.h>

PYBIND11_MODULE(_C, m) {
    BIND_TK_PARALLEL_TENSOR(m);
    m.def("tk_all_to_all", &entrypoint);
}
