
# Name of project
project=raytracer

# Compiler to use
cc=nvcc 
# Location to store object files
DO=obj
# Directory for main binaries
DB=bin
# Directory where source files are
DS=src
# Directory where docs are stored
DD=doc

# NVIDIA COMPILER FLAGS
FNV=-Wno-deprecated-gpu-targets 
#FNV=-c -O2 --compiler-bindir /usr/bin
# Compiler flags to use for debugging
FD=$(FNV) -Wall -g 
# Compiler flags to use for object files
FO=$(FNV) -c
# Compiler Flags to use for binaries
FB=$(FNV) -lGL -lGLU -lglut -lcudadevrt
# Link CUDA object files
LC=-dc

################################################
# Build Commands
################################################

all: prep $(project)

# Remove any previously built files
clean:
	# Remove any objects from the object directory
	rm -rf $(DO)

# Removes any files other thatn the source from the directory
purge: clean
	#Remove any binaries from the output directory
	rm -rf $(DB)
	#Remove the source tarball if it exists
	rm -rf $(project).tar.gz
	#Remove the documentation files
	rm -rf $(DD)

# Create the directory structure required for output
prep:
	# Create the objects directory
	mkdir -p $(DO)
	# Create output directory
	mkdir -p $(DB)

# Create the documentation for the project
documentation:
	#Generating documentaton
	doxygen Doxyfile

# Tar the project to make it more portable
tarball:
	# Creating tarball
	tar -zcv -f $(project).tar.gz Makefile src README{.src,,.txt,.md} Doxyfile


################################################
# Executable Binaries
################################################

#Build project executable
$(project): prep driver.o light.o object.o material.o sphere.o triangle.o world.o vector3.o camera.o raytracer.o frame.o canvas.o helpers.o
	# Building and linking the project binary
	$(cc) -rdc=true -o $(DB)/$@ $(DO)/driver.o $(DO)/light.o $(DO)/object.o $(DO)/material.o $(DO)/sphere.o $(DO)/triangle.o $(DO)/world.o $(DO)/vector3.o $(DO)/camera.o $(DO)/raytracer.o $(DO)/frame.o $(DO)/canvas.o $(DO)/helpers.o $(FB)

################################################
# Object Files
################################################

driver.o: $(DS)/driver.cu
	# Compiling driver object
	$(cc) $(FO) -o $(DO)/$@ $^

light.o: $(DS)/light.cu
	# Compiling light object
	$(cc) $(FO) -o $(DO)/$@ $^

object.o: $(DS)/object.cu
	# Compiling object object
	$(cc) $(FO) $(LC) -o $(DO)/$@ $^

material.o: $(DS)/material.cu
	# Compiling material object
	$(cc) $(FO) -o $(DO)/$@ $^

sphere.o: $(DS)/sphere.cu
	# Compiling sphere object
	$(cc) $(FO) $(LC) -o $(DO)/$@ $^

triangle.o: $(DS)/triangle.cu
	# Compiling triangle object
	$(cc) $(FO) $(LC) -o $(DO)/$@ $^

world.o: $(DS)/world.cu
	# Compiling world object
	$(cc) $(FO) -o $(DO)/$@ $^

vector3.o: $(DS)/vector3.cu
	# Compiling vector object
	$(cc) $(FO) $(LC) -o $(DO)/$@ $^

camera.o: $(DS)/camera.cu
	# Compiling camera object
	$(cc) $(FO) -o $(DO)/$@ $^

raytracer.o: $(DS)/raytracer.cu
	# Compiling raytracer object
	$(cc) $(FO) $(LC) -o $(DO)/$@ $^

frame.o: $(DS)/frame.cu
	# Compiling frame object
	$(cc) $(FO) -o $(DO)/$@ $^

canvas.o: $(DS)/canvas.cu
	# Compiling canvas object
	$(cc) $(FO) -o $(DO)/$@ $^

helpers.o: $(DS)/helpers.cu
	# Compiling helpers object
	$(cc) $(FO) $(LC) -o $(DO)/$@ $^

