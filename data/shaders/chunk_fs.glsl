#version 330
#extension GL_ARB_gpu_shader5 : enable
precision highp float;

uniform sampler1D colorPalette;
uniform sampler2D cells;

in  vec2 texcoord;			// interpolated cell coordinates
out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

// Size of the chunk in cells
int chunkSize = 128;
int paletteSize = 32 * 64;

float pos1d(vec2 v) {
	float _x = floor(v.x * 64) / paletteSize;
	float _y = floor(v.y * 32) / 32;

	return _y + _x;
}

void main() {
	// Read the cell data from the chunk texture
	vec4 cellData = texture(cells, texcoord * (chunkSize - 1) / chunkSize);

	int byte = int(cellData.r * 255.0);
	int bytes = int(cellData.r * 255.0 + cellData.g * 255.0 * 255.0);

	unsigned int type = (byte & 0x001F);
	unsigned int shade = (bytes & 0x07E0) / 32;

	vec4 color = texture(colorPalette, type / 32.0 + shade / (32.0 * 64.0));
	
	fragmentColor = color;
}
