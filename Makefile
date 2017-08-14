CPP     	= 	g++ -O3  -std=c++1z -fexceptions -D__GLIBCXX_TYPE_INT_N_0=__int128 -D__GLIBCXX_BITSIZE_INT_N_0=128
STRIPPER 	= 	./removecomments.pl
EMCC    	=       ~/Extern/emscripten3/emscripten/emcc -Wc++11-extensions -std=c++11
OPT             = 	 -s PRECISE_I64_MATH=1
SET     	= 	-O2 -s ALLOW_MEMORY_GROWTH=1 -s ASM_JS=0
LIBPATH 	= 
SYSLIBS 	= 	-lstdc++ -lboost_regex -lboost_thread -lboost_system -lboost_date_time -lrt  -lpthread -lm
PRGS		=	st1tch3r


all: le_build st1tch3r

st1tch3r: le_build/st1tch3r.o
	$(CPP) -fexceptions -D_BOOL  $^ -I$(INCPATH) -L$(LIBPATH) $(SYSLIBS) `pkg-config --libs opencv` -o $@
le_build/st1tch3r.o: st1tch3r.cpp
	$(CPP) -fexceptions -D_BOOL -c $< -I$(INCPATH) -L$(LIBPATH)  -DHAVE_IOMANIP -DHAVE_IOSTREAM -DHAVE_LIMITS_H -o $@


clean:
	rm -rf le_build findiffd;\


le_build:
	mkdir ./le_build



