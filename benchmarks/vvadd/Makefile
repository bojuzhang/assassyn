CC=riscv64-unknown-elf-gcc
CFLAGS=-march=rv32i -mabi=ilp32 -O0 -g -mno-relax -I.
LDFLAGS=-nostdlib -nostartfiles -Ttext 0x0
SRC=vvadd_main.c
ASM_SRC=../../entry.S
OBJ=entry.o $(SRC:.c=.o)
TARGET=vvadd.elf

all: $(TARGET)

entry.o: $(ASM_SRC)
	$(CC) $(CFLAGS) -c $< -o $@

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(OBJ) -lgcc
	riscv64-unknown-elf-objcopy -O binary $@ vvadd.bin
	riscv64-unknown-elf-objdump -d $@ > vvadd.dump

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o *.elf *.bin *.dump
