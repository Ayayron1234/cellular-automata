#pragma once
#include <SDL2/SDL.h>
#undef main

#include "gl/glew.h"
#include "gl/GL.h"
#include <gl/glu.h>

#if defined(IMGUI_IMPL_OPENGL_ES2)
#include <SDL2/SDL_opengles2.h>
#else
#include <SDL2/SDL_opengl.h>
#endif
