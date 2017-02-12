CC := g++ 
CFLAGS := -Wall -Wextra -pedantic
CFLAGS +=-I./include/maskedcnn

SRC := src/Activation.cpp src/Layer.cpp src/Network.cpp
OBJ := $(patsubst %.c, %.o, $(SRC))


MaskedCNN: $(OBJ)
	$(CC) $(CFLAGS) $(OBJ) -o MaskedCNN


%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@
clean:
	rm -f ./src/*.o MaskedCNN
