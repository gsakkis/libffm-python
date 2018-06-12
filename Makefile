.PHONY: all libffm clean

CXX = g++
CXXFLAGS = -Wall -O3 -std=c++0x -march=native

# comment the following flags if you do not want to SSE instructions
DFLAG += -DUSESSE

# comment the following flags if you do not want to use OpenMP
DFLAG += -DUSEOMP
CXXFLAGS += -fopenmp

all: libffm ffm/libffm.so

libffm:
	$(MAKE) -C libffm

ffm/libffm.so: ffm-wrapper.cpp ffm-wrapper.h libffm/timer.cpp
	$(CXX) -shared $(CXXFLAGS) $(DFLAG) -o $@ -fPIC libffm/timer.cpp $<

clean:
	$(MAKE) -C libffm clean
	rm -rf libffm.so build/ dist/ ffm.egg-info/
