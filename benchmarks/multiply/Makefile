CC=riscv64-unknown-elf-gcc
CFLAGS=-march=rv32i -mabi=ilp32 -O0 -g -mno-relax
LDFLAGS=-nostdlib -nostartfiles -T link.ld
SRC=$(wildcard *.c)
ASM_SRC=../../entry.S
OBJ=entry.o $(SRC:.c=.o)
TARGET=multiply.elf

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ -lgcc
	riscv64-unknown-elf-objcopy -O binary $@ multiply.bin
	riscv64-unknown-elf-objdump -d $@ > multiply.dump

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

entry.o: $(ASM_SRC)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET) multiply.bin multiply.dump multiply.hex
