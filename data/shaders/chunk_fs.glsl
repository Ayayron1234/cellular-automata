#version 330
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

	float type = (int(cellData.r * 255.0) << 3) / 255.0;
	float shade = (int(cellData.r * 255.0 * 255.0 + cellData.g * 255.0) >> 5) / 65535.0;

	vec4 color = texture(colorPalette, pos1d(vec2(shade, type)));
	
	fragmentColor = color;
}
