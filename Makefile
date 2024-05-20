CC = gcc
RM = rm -f

all: clean cogsnet-compute

cogsnet-compute:
	$(CC) -o cogsnet-compute.o cogsnet-compute.c -lm

clean:
	$(RM) cogsnet-compute.o
