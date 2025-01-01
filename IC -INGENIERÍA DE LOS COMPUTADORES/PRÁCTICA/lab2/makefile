CC = mpicc
CFLAGS = -Wall -O2
LDFLAGS = -L. -lprocessing

# Nombre del ejecutable
TARGET = main_mpi.elf

# Archivos fuente
SRCS = main_mpi.c

# Objetos generados
OBJS = $(SRCS:.c=.o)

# Regla por defecto
all: $(TARGET)

# Regla para compilar el programa
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Regla para compilar archivos .c a .o
%.o: %.c
	$(CC) $(CFLAGS) -c $<

# Regla para limpiar archivos generados
clean:
	rm -f $(OBJS) $(TARGET)

# Ejercucón del programa
run:
	mpirun -np 4 ./$(TARGET)

.PHONY: all clean run

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:.