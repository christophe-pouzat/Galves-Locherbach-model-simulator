P=programe_name
OBJECTS=
CFLAGS += `pkg-config --cflags glib-2.0 gsl` -g -Wall -O3 -std=gnu11 
LDLIBS = `pkg-config --libs glib-2.0 gsl ` 

$(P): $(OBJECTS)
