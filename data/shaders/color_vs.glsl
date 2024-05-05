#version 330
precision highp float;

layout(location = 0) in vec2 vertPos;	// Attrib Array 0
layout(location = 1) in vec4 vertColor;	// Attrib Array 1

struct Camera {
	vec2 position;
	float zoom;
	float aspectRatio;
};
uniform Camera camera;

out vec4 color;

void main() {
	color = vertColor;																												// -1,1 to 0,1
	gl_Position = vec4(((vertPos.xy + camera.position) * camera.zoom) * vec2(1.0 / camera.aspectRatio, 1.0), 0, 1); 				// transform to clipping space
}
