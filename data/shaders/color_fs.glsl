#version 330
precision highp float;

in  vec4 color;				// interpolated cell coordinates
out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

void main() {
	fragmentColor = color;
}
