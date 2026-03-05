#pragma once
#include "common.h"

PipelineResult run_pipeline(const PipelineConfig& config, const char* kernel_path);

// GPU-only diagnostic: compare clFinish vs event-poll overhead
void run_gpu_diagnostic(int hidden_dim, int num_steps, const char* kernel_path);
