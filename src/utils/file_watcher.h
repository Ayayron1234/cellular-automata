#pragma once
#include <iostream>
#include <fstream>
#include <filesystem>

class TextFileResource {
public:
	TextFileResource(const std::string& path)
		: m_path(path)
	{
		load();
	}

	bool updated() const {
		return m_lastModified != std::filesystem::last_write_time(m_path);
	}

	void load() {
		m_ifs.open(m_path);

		// Load value and set last write time
		m_value = std::string((std::istreambuf_iterator<char>(m_ifs)), std::istreambuf_iterator<char>());
		m_lastModified = std::filesystem::last_write_time(m_path);

		m_ifs.close();
	}

	const std::string& value() const {
		return m_value;
	}

private:
	using time_type = std::filesystem::file_time_type;

	std::string			m_value;

	const std::string	m_path;
	std::ifstream		m_ifs;
	time_type			m_lastModified;

};
