## CC COMPILER OPTIONS ##

# CC compiler options:
CC=clang++
CC_FLAGS=-g -std=c++17 -O3
CC_LIBS=

## Project file structure ##

# Source file directory:
SRC_DIR = src

# Object file directory:
OBJ_DIR = bin

# Include header file diretory:
INC_DIR = include

##########################################################

## Make variables ##

# Target executable name:
EXE = sequential_sudoku_solver

# Object files:
OBJS = $(OBJ_DIR)/main.o 
OBJS += $(OBJ_DIR)/sequential_solver.o 
OBJS += $(OBJ_DIR)/board.o

##########################################################

## Compile ##

# Link c++ compiled object files to target executable:
$(EXE) : $(OBJS)
	clang-format -i ./include/* ./src/*; $(CC) $(CC_FLAGS) $(OBJS) -o $@

# Compile main .cpp file to object files:
$(OBJ_DIR)/%.o : %.cpp
	 $(CC) $(CC_FLAGS) -c $< -o $@

# Compile C++ source files to object files:
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cpp include/%.hpp
	$(CC) $(CC_FLAGS) -c $< -o $@

# Clean objects in object directory.
clean:
	$(RM) bin/* *.o $(EXE)
