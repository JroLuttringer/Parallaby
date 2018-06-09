CC=mpicc
CFLAGS = -O2 -g -Wall -I /usr/X11R6/include
LDLIBS = -L /usr/X11R6/lib -lX11

ALL=chemin_lab gen_lab_mpi gen_lab

all: $(ALL)

$(ALL): graph.o

clean:
	/bin/rm -f $(ALL) *.o laby.lab
