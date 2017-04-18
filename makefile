
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
FO=$(FNV)-c 
# Compiler Flags to use for binaries
FB=$(FNV) 

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

# Tar the project to make it easier to move around
tarball:
	# Creating tarball
	tar -zcv -f $(project).tar.gz makefile src README{.src,,.txt,.md} {D,d}oxyfile


################################################
# Executable Binaries
################################################

#Build project executable
$(project): prep driver.o object.o light.o sphere.o plane.o
	# Building and linking the project binary
	$(cc) -o $(DB)/$@ $(DO)/driver.o $(DO)/object.o $(DO)/light.o $(DO)/sphere.o $(DO)/plane.o $(FB)

################################################
# Object Files
################################################

driver.o: $(DS)/driver.cu
	# Compiling driver object
	$(cc) $(FO) -o $(DO)/$@ $^

object.o: $(DS)/object.cu
	# Compiling object object
	$(cc) $(FO) -o $(DO)/$@ $^

light.o: $(DS)/light.cu
	# Compiling light object
	$(cc) $(FO) -o $(DO)/$@ $^

sphere.o: $(DS)/sphere.cu
	# Compiling sphere object
	$(cc) $(FO) -o $(DO)/$@ $^

plane.o: $(DS)/plane.cu
	# Compiling plane object
	$(cc) $(FO) -o $(DO)/$@ $^

