# Compiler y flags
CXX = g++
CXXFLAGS = -O3 -fopenmp -std=c++17

# Targets
all: tarea3 tarea4 tarea5

tarea3: tarea3.cpp
	$(CXX) $(CXXFLAGS) -o tarea3 tarea3.cpp

tarea4: tarea4.cpp
	$(CXX) $(CXXFLAGS) -o tarea4 tarea4.cpp

tarea5: tarea5.cpp
	$(CXX) $(CXXFLAGS) -o tarea5 tarea5.cpp

clean:
	rm -f tarea3 tarea4 tarea5

.PHONY: all clean