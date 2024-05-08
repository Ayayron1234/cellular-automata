#pragma once
#include <iostream>
#include <fstream>
#include <filesystem>

class FileResource {
public:
	FileResource(const std::string& path)
		: m_path(path)
	{ }

	FileResource(const FileResource& resource) 
		: m_path(resource.m_path)
		, m_lastModified(resource.m_lastModified)
	{ }

	bool updated() const {
		return m_swapped 
			|| m_lastModified != std::filesystem::last_write_time(m_path);
	}

	void swap(const std::string& path) {
		m_path = path;
		m_swapped = true;
	}

protected:
	using time_type = std::filesystem::file_time_type;

	std::string	m_path;
	time_type			m_lastModified;
	bool				m_swapped = false;

	void loaded() {
		m_lastModified = std::filesystem::last_write_time(m_path);
		m_swapped = false;
	}

};

class TextFileResource : public FileResource {
public:
	TextFileResource(const std::string& path)
		: FileResource(path)
	{
		load();
		loaded();
	}

	void load() {
		m_ifs.open(m_path);

		// Load value and set last write time
		m_value = std::string((std::istreambuf_iterator<char>(m_ifs)), std::istreambuf_iterator<char>());
		loaded();

		m_ifs.close();
	}

	const std::string& value() const {
		return m_value;
	}

private:
	std::ifstream		m_ifs;
	std::string			m_value;

};

class BitmapFileResource : public FileResource {
public:
	BitmapFileResource(const std::string& path)
		: FileResource(path)
	{
		load();
		loaded();
	}

	BitmapFileResource(const BitmapFileResource& bm) 
		: FileResource(bm)
		, m_value(bm.m_value)
	{ }

	void load() {
		m_value = std::make_shared<Bitmap>(m_path.c_str());
		loaded();
	}

	const Bitmap& value() const {
		return *m_value;
	}

private:
	std::shared_ptr<Bitmap> m_value;

};

class JsonFileResource : public FileResource {
public:
	JsonFileResource(const std::string& path)
		: FileResource(path)
	{
		load();
		loaded();
	}

	void load() {
		m_ifs.open(m_path);

		m_ifs >> m_value;
		loaded();

		m_ifs.close();
	}

	const Json& value() const {
		return m_value;
	}

private:
	std::ifstream	m_ifs;
	Json			m_value;

};
