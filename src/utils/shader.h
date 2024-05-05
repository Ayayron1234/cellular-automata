#pragma once
#include <iostream>
#include <string>
#include "../graphics_headers.h"
#include "vec2.h"
#include "file_watcher.h"


class ShaderSourceFile {
public:
	ShaderSourceFile(const char* path) {
		if (path == nullptr)
			return;

		std::ifstream file(path);
		if (!file.is_open()) {
			std::cerr << "Failed to open shader source file: " << path << std::endl;
			return;
		}

		m_source = std::make_unique<std::string>((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
	}

	const char* getSource() {
		if (!m_source)
			return nullptr;

		return m_source->c_str();
	}

private:
	std::unique_ptr<std::string> m_source;
};

class GPUProgram {
	//--------------------------
	unsigned int shaderProgramId = 0;
	unsigned int vertexShader = 0, geometryShader = 0, fragmentShader = 0;
	bool waitError = true;

	void getErrorInfo(unsigned int handle) { // shader error report
		int logLen, written;
		glGetShaderiv(handle, GL_INFO_LOG_LENGTH, &logLen);
		if (logLen > 0) {
			std::string log(logLen, '\0');
			glGetShaderInfoLog(handle, logLen, &written, &log[0]);
			printf("Shader log:\n%s", log.c_str());
			if (waitError) getchar();
		}
	}

	bool checkShader(unsigned int shader, std::string message) { // check if shader could be compiled
		int OK;
		glGetShaderiv(shader, GL_COMPILE_STATUS, &OK);
		if (!OK) {
			printf("%s!\n", message.c_str());
			getErrorInfo(shader);
			return false;
		}
		return true;
	}

	bool checkLinking(unsigned int program) { 	// check if shader could be linked
		int OK;
		glGetProgramiv(program, GL_LINK_STATUS, &OK);
		if (!OK) {
			printf("Failed to link shader program!\n");
			getErrorInfo(program);
			return false;
		}
		return true;
	}

	int getLocation(const std::string& name) {	// get the address of a GPU uniform variable
		int location = glGetUniformLocation(shaderProgramId, name.c_str());
		if (location < 0) printf("uniform %s cannot be set\n", name.c_str());
		return location;
	}

public:
	GPUProgram(bool _waitError = true) { shaderProgramId = 0; waitError = _waitError; }

	GPUProgram(const GPUProgram& program) = default;
	GPUProgram& operator=(const GPUProgram& program) = default;

	unsigned int getId() { return shaderProgramId; }

	static GPUProgram loadFromFiles(const char* vertexPath, const char* fragmentPath, const char* fragmentShaderOutputName, const char* geometryPath = nullptr) {
		ShaderSourceFile vertSource(vertexPath);
		ShaderSourceFile fragSource(fragmentPath);
		ShaderSourceFile geomSource(geometryPath);

		GPUProgram program;
		program.create(vertSource.getSource(), fragSource.getSource(), fragmentShaderOutputName, geomSource.getSource());
		return program;
	}

	bool create(const char* const vertexShaderSource,
		const char* const fragmentShaderSource, const char* const fragmentShaderOutputName,
		const char* const geometryShaderSource = nullptr)
	{
		// Create vertex shader from string
		if (vertexShader == 0) vertexShader = glCreateShader(GL_VERTEX_SHADER);
		if (!vertexShader) {
			printf("Error in vertex shader creation\n");
			exit(1);
		}
		glShaderSource(vertexShader, 1, (const GLchar**)&vertexShaderSource, NULL);
		glCompileShader(vertexShader);
		if (!checkShader(vertexShader, "Vertex shader error")) return false;

		// Create geometry shader from string if given
		if (geometryShaderSource != nullptr) {
			if (geometryShader == 0) geometryShader = glCreateShader(GL_GEOMETRY_SHADER);
			if (!geometryShader) {
				printf("Error in geometry shader creation\n");
				exit(1);
			}
			glShaderSource(geometryShader, 1, (const GLchar**)&geometryShaderSource, NULL);
			glCompileShader(geometryShader);
			if (!checkShader(geometryShader, "Geometry shader error")) return false;
		}

		// Create fragment shader from string
		if (fragmentShader == 0) fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
		if (!fragmentShader) {
			printf("Error in fragment shader creation\n");
			exit(1);
		}

		glShaderSource(fragmentShader, 1, (const GLchar**)&fragmentShaderSource, NULL);
		glCompileShader(fragmentShader);
		if (!checkShader(fragmentShader, "Fragment shader error")) return false;

		shaderProgramId = glCreateProgram();
		if (!shaderProgramId) {
			printf("Error in shader program creation\n");
			exit(1);
		}
		glAttachShader(shaderProgramId, vertexShader);
		glAttachShader(shaderProgramId, fragmentShader);
		if (geometryShader > 0) glAttachShader(shaderProgramId, geometryShader);

		// Connect the fragmentColor to the frame buffer memory
		glBindFragDataLocation(shaderProgramId, 0, fragmentShaderOutputName);	// this output goes to the frame buffer memory

		// program packaging
		glLinkProgram(shaderProgramId);
		if (!checkLinking(shaderProgramId)) return false;

		// make this program run
		glUseProgram(shaderProgramId);
		return true;
	}

	void Use() { 		// make this program run
		glUseProgram(shaderProgramId);
	}

	void setUniform(int i, const std::string& name) {
		int location = getLocation(name);
		if (location >= 0) glUniform1i(location, i);
	}

	void setUniform(float f, const std::string& name) {
		int location = getLocation(name);
		if (location >= 0) glUniform1f(location, f);
	}

	void setUniform(const vec2& v, const std::string& name) {
		int location = getLocation(name);
		GLfloat _v[2] = { v.x, v.y };
		if (location >= 0) glUniform2fv(location, 1, &_v[0]);
	}

	~GPUProgram() { if (shaderProgramId > 0) glDeleteProgram(shaderProgramId); }
};

class Shader {
public:
	Shader() { }

	void setSourceFiles(const std::string& vertPath, const std::string& fragPath) {
		m_vertexSource		= std::make_unique<TextFileResource>(vertPath);
		m_fragmentSource	= std::make_unique<TextFileResource>(fragPath);
	}

	void setSourceFiles(const std::string& vertPath, const std::string& fragPath, const std::string& geomPath) {
		m_vertexSource		= std::make_unique<TextFileResource>(vertPath);
		m_fragmentSource	= std::make_unique<TextFileResource>(fragPath);
		m_geometrySource	= std::make_unique<TextFileResource>(geomPath);
	}

	void create() {
		compileShaders();
		createProgram();
		attachAndLinkShaders();
	}

	void recompileIfSourceChanged() {
		bool didRecompile = false;
		didRecompile |= recompileShaderIfUpdated(m_vertexShader, m_vertexSource, GL_VERTEX_SHADER);
		didRecompile |= recompileShaderIfUpdated(m_fragmentShader, m_fragmentSource, GL_FRAGMENT_SHADER);

		if (m_geometrySource)
			didRecompile |= recompileShaderIfUpdated(m_geometryShader, m_geometrySource, GL_GEOMETRY_SHADER);

		if (didRecompile)
			attachAndLinkShaders();
	}

	GLuint getID() const {
		return m_programID;
	}

	void use() {
		glUseProgram(getID());
	}

private:
	using ShaderSource = std::unique_ptr<TextFileResource>;

	GLuint m_programID = 0;
	
	GLuint			m_vertexShader = 0;
	ShaderSource	m_vertexSource;

	GLuint			m_fragmentShader = 0;
	ShaderSource	m_fragmentSource;

	GLuint			m_geometryShader = 0; 
	ShaderSource	m_geometrySource;

	bool recompileShaderIfUpdated(GLuint shader, const ShaderSource& source, int type) {
		if (source->updated()) {
			std::string prevShaderSource = source->value();

			std::cout << "Recompiling shader" << std::endl;
			source->load();

			return compileShader(&shader, source->value().c_str(), type);
		}
		return false;
	}

	void compileShaders() {
		compileShader(&m_vertexShader, m_vertexSource->value().c_str(), GL_VERTEX_SHADER);
		compileShader(&m_fragmentShader, m_fragmentSource->value().c_str(), GL_FRAGMENT_SHADER);

		if (m_geometrySource)
			compileShader(&m_geometryShader, m_geometrySource->value().c_str(), GL_GEOMETRY_SHADER);
	}

	void createProgram() {
		// Create program
		m_programID = glCreateProgram();
		if (!getID()) {
			std::cerr << "Error in shader program creation" << std::endl;
			exit(1);
		}
	}

	void attachAndLinkShaders() {
		// Attach shaders
		glAttachShader(getID(), m_vertexShader);
		glAttachShader(getID(), m_fragmentShader);
		if (m_geometryShader > 0)
			glAttachShader(getID(), m_geometryShader);

		// Connect the fragmentColor to the frame buffer memory
		glBindFragDataLocation(getID(), 0, "outColor");

		// program packaging
		glLinkProgram(getID());
		if (!checkLinking(getID()))
			exit(1);

		// make this program run
		glUseProgram(getID());
	}

	bool compileShader(GLuint* shader, const char* shaderSource, int shaderType) {
		// Create shader if not created yet
		if (*shader == 0) 
			*shader = glCreateShader(shaderType);

		// Check for errors
		if (!*shader) {
			std::cerr << "Error creating vertex shader" << std::endl;
			exit(1);
		}

		// Compile shader
		glShaderSource(*shader, 1, (const GLchar**)&shaderSource, NULL);
		glCompileShader(*shader);
		
		// Return compilation status
		return checkShaderCompilation(*shader);
	}

	bool checkShaderCompilation(GLuint shader) {
		int OK;
		glGetShaderiv(shader, GL_COMPILE_STATUS, &OK);
		if (!OK) {
			std::cerr << "Failed to compile shader!" << std::endl;
			writeShaderCompilationErrorInfo(shader);
			return false;
		}
		return true;
	}

	void writeShaderCompilationErrorInfo(unsigned int handle) {
		int logLen, written;
		glGetShaderiv(handle, GL_INFO_LOG_LENGTH, &logLen);

		if (logLen > 0) {
			std::string log(logLen, '\0');
			glGetShaderInfoLog(handle, logLen, &written, &log[0]);

			std::cout << "Shader log:\n" << log << std::endl;
		}
	}

	bool checkLinking(unsigned int program) { 
		int OK;
		glGetProgramiv(program, GL_LINK_STATUS, &OK);
		if (!OK) {
			std::cerr << "Failed to link shader program!" << std::endl;
			writeShaderCompilationErrorInfo(program);
			return false;
		}
		return true;
	}

};
