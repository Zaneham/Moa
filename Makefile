# Makefile -- Moa Monte Carlo neutron transport
#
# Build with MinGW GCC on Windows (Git Bash).
# No autoconf, no cmake, no drama. Just make.

CC       = gcc
WARNS    = -Wall -Wextra -Wpedantic -Wshadow -Wconversion \
           -Wdouble-promotion -Wformat=2 -Wundef \
           -Wstrict-prototypes -Wmissing-prototypes \
           -Wno-sign-conversion
CFLAGS   = -std=c99 $(WARNS) -O2 -I.
TFLAGS   = -std=c99 $(WARNS) -O0 -g -I.
LDFLAGS  = -lm

# Kauri path
KAURI    = C:/dev/kauri

# OpenMP (set OMP=1 to enable: make OMP=1)
OMP     ?= 0
ifeq ($(OMP),1)
CFLAGS  += -fopenmp
TFLAGS  += -fopenmp
LDFLAGS += -fopenmp
endif

# GPU via BarraCUDA
# GPU=1  — AMD HSA backend (Linux/ROCm only, bc_runtime)
# GPU=NV — NVIDIA CUDA Driver API backend (Windows/Linux, nv_rt)
GPU     ?= 0
BCSRC    = C:/dev/compilers/barracuda
ifeq ($(GPU),1)
CFLAGS  += -DMOA_GPU -I$(BCSRC)/src/runtime
TFLAGS  += -DMOA_GPU -I$(BCSRC)/src/runtime
GPU_OBJ  = gpu/gp_host.o gpu/bc_runtime.o gpu/bc_abend.o
LDFLAGS += -ldl
endif
ifeq ($(GPU),NV)
CFLAGS  += -DMOA_GPU -DMOA_GPU_NV -I$(BCSRC)/src/nvidia
TFLAGS  += -DMOA_GPU -DMOA_GPU_NV -I$(BCSRC)/src/nvidia
GPU_OBJ  = gpu/gp_nv.o gpu/nv_rt.o
endif

# Sources
SRC      = src/rng.c src/nd_parse.c src/nd_xs.c src/nd_res.c src/nd_rmat.c \
           src/nd_urr.c src/nd_dopl.c src/nd_thrm.c src/nd_sab.c \
           src/csg.c src/cg_lat.c \
           src/tl_score.c src/tl_ebin.c src/tl_mesh.c \
           src/tp_loop.c src/tp_crit.c src/tp_fixd.c src/io_input.c
MAIN_SRC = src/main.c
TEST_SRC = tests/tmain.c tests/trng.c tests/tparse.c tests/txs.c \
           tests/tgeom.c tests/ttrans.c tests/tnew.c

# Objects (release)
OBJ      = $(SRC:.c=.o)
MAIN_OBJ = $(MAIN_SRC:.c=.o)

# Objects (test — built in tests/obj/ to avoid clobbering release .o)
TEST_SOBJ = $(patsubst src/%.c, tests/obj/%.o, $(SRC))
TEST_TOBJ = $(patsubst tests/%.c, tests/obj/%.o, $(TEST_SRC))

# Targets
BIN      = moa.exe
TEST_BIN = moa_test.exe

.PHONY: all test clean

all: $(BIN)

ifneq ($(GPU_OBJ),)
$(BIN): $(OBJ) $(MAIN_OBJ) $(GPU_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
else
$(BIN): $(OBJ) $(MAIN_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
endif

tests/obj:
	mkdir -p tests/obj

# Test objects: src/*.c compiled with TFLAGS (no optimisation, debug)
tests/obj/%.o: src/%.c moa.h | tests/obj
	$(CC) $(TFLAGS) -c -o $@ $<

# Test objects: tests/*.c compiled with TFLAGS
tests/obj/%.o: tests/%.c moa.h tests/tharns.h | tests/obj
	$(CC) $(TFLAGS) -c -o $@ $<

$(TEST_BIN): $(TEST_SOBJ) $(TEST_TOBJ)
	$(CC) $(TFLAGS) -o $@ $^ $(LDFLAGS)

test: $(TEST_BIN)
	./$(TEST_BIN)

# Compile rules (release)
src/%.o: src/%.c moa.h
	$(CC) $(CFLAGS) -c -o $@ $<

# GPU compile rules (GPU=1 only)
ifeq ($(GPU),1)
gpu/gp_host.o: gpu/gp_host.c gpu/gp_host.h moa.h
	$(CC) $(CFLAGS) -c -o $@ $<

gpu/bc_runtime.o: $(BCSRC)/src/runtime/bc_runtime.c $(BCSRC)/src/runtime/bc_runtime.h
	$(CC) $(CFLAGS) -Wno-switch-enum -c -o $@ $<

gpu/bc_abend.o: $(BCSRC)/src/runtime/bc_abend.c $(BCSRC)/src/runtime/bc_abend.h
	$(CC) $(CFLAGS) -c -o $@ $<

gpu/tp_kern.hsaco: gpu/tp_kern.cu
	$(BCSRC)/barracuda --amdgpu-bin $< -o $@

.PHONY: gpu
gpu: gpu/tp_kern.hsaco
endif

# GPU compile rules (GPU=NV only)
ifeq ($(GPU),NV)
gpu/gp_nv.o: gpu/gp_nv.c gpu/gp_nv.h moa.h
	$(CC) $(CFLAGS) -c -o $@ $<

gpu/nv_rt.o: $(BCSRC)/src/nvidia/nv_rt.c $(BCSRC)/src/nvidia/nv_rt.h
	$(CC) $(CFLAGS) -c -o $@ $<

gpu/tp_kern.ptx: gpu/tp_kern.cu
	$(BCSRC)/barracuda --nvidia-ptx $< -o $@

.PHONY: gpu
gpu: gpu/tp_kern.ptx
endif

clean:
	rm -f src/*.o $(BIN) $(TEST_BIN) gpu/*.o gpu/*.hsaco gpu/*.ptx
	rm -rf tests/obj
