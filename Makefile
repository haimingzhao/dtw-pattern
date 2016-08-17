BUILDDIR := build
PRODUCT := main

SRCDIR := src

HDRS := $(wildcard $(SRCDIR)/*.h)

CSRCS := $(wildcard $(SRCDIR)/*.C)
CPPSRCS := $(wildcard $(SRCDIR)/*.cpp)
FSRCS += $(wildcard $(SRCDIR)/fortran/*.f90)
CUSRCS += $(wildcard $(SRCDIR)/*.cu)

OBJS := $(CSRCS:$(SRCDIR)/%.C=$(BUILDDIR)/%.o)
OBJS := $(CPPSRCS:$(SRCDIR)/%.cpp=$(BUILDDIR)/%.o)
OBJS += $(FSRCS:$(SRCDIR)/fortran/%.f90=$(BUILDDIR)/%.o)
OBJS += $(CUSRCS:$(SRCDIR)/%.cu=$(BUILDDIR)/%.o)

BINARY := $(BUILDDIR)/$(PRODUCT)

CUDA_DIR := /local/java/cuda

# gcc flags:
CXX := g++
CXXFLAGS_DEBUG := -g -DDEBUG -std=c++11
CXXFLAGS_TIME := -DTIME
CXXFLAGS_OPT := -O0
CXXFLAGS_OPENMP := -fopenmp

FC := gfortran
FFLAGS := -ffree-form

NVCC := nvcc
NVFLAGS := -O0 -g -G -gencode arch=compute_35,code=compute_35
NVFLAGS := $(CXXFLAGS_TIME) $(NVFLAGS)

LD := $(CXX)

# select optimized or debug
# CXXFLAGS := $(CXXFLAGS_OPT) $(CPPFLAGS) -I$(CUDA_DIR)/include
# CXXFLAGS := $(CXXFLAGS_OPT) $(CXXFLAGS_TIME) $(CPPFLAGS) -I$(CUDA_DIR)/include
CXXFLAGS := $(CXXFLAGS_OPT) $(CXXFLAGS_DEBUG) $(CXXFLAGS_TIME) $(CPPFLAGS) -I$(CUDA_DIR)/include

# include paths
OS := $(shell uname)
ifeq ($(OS), Darwin)
# Run MacOS commands
    CXXINCLUDES:= -I/usr/local/Cellar/boost/1.59.0/include/
else
# check for Linux and run other commands
endif

# add openmp flags (comment out for serial build)
#CXXFLAGS += $(CXXFLAGS_OPENMP)
#LDFLAGS += $(CXXFLAGS_OPENMP)

LDFLAGS +=-L$(CUDA_DIR)/lib64 -lcudart

all : $(BINARY)

$(BINARY) : $(OBJS)
	@echo linking $@
	$(maketargetdir)
	$(LD) $(LDFLAGS) -o $@ $^

$(BUILDDIR)/%.o : $(SRCDIR)/%.C
	@echo compiling $<
	$(maketargetdir)
	$(CXX) $(CXXFLAGS) $(CXXINCLUDES) -c -o $@ $<

$(BUILDDIR)/%.o : $(SRCDIR)/%.cpp
	@echo compiling $<
	$(maketargetdir)
	$(CXX) $(CXXFLAGS) $(CXXINCLUDES) -c -o $@ $<

$(BUILDDIR)/%.o : $(SRCDIR)/fortran/%.f90
	@echo compiling $<
	$(maketargetdir)
	$(FC) $(FFLAGS) -c -o $@ $<

$(BUILDDIR)/%.o : $(SRCDIR)/%.cu
	@echo compiling $<
	$(maketargetdir)
	$(NVCC) $(NVFLAGS) -c -o $@ $<

define maketargetdir
	-@mkdir -p $(dir $@) > /dev/null 2>&1
endef

clean :
	rm -f $(BINARY) $(OBJS)
	rm -rf $(BUILDDIR)
