#pragma once
#include "../simulation.h"
#include <filesystem>
#include <vector>
#include <tuple>
#include "../utils.h"

class Brush {
public:
    enum class Shape { Square = 0x00, Circle };

    Brush() = default;

    Brush(Cell::Type type)
        : m_type(type)
    { }

	void draw(const CellCoord& coord, World& world) {
        // Iterate over cells in a r * r square
        for (int dX = -m_radius; dX < m_radius; ++dX)
            for (int dY = -m_radius; dY < m_radius; ++dY) {
                CellCoord subCoord = CellCoord{ coord.x + dX, coord.y + dY };
                
                // Dont draw to the corners when the bush shape is a circle
                if (m_shape == Shape::Circle) {
                    float distance = length(vec2(dX, dY));
                    if (distance > m_radius)
                        continue;
                }

                // Create cell and set even tick
                Cell cell = getCell(subCoord);
                cell.updatedOnEvenTick = world.evenTick();

                // Set cell in world
                world.setCell(subCoord, cell);
            }
	}

    void setType(Cell::Type type) {
        m_type = type;
    }

    virtual Brush* showImGuiWidget(Options& options) { return this; }

    void setRadius(int radius) {
        m_radius = radius;
    }

    void setShape(const Shape& shape) {
        m_shape = shape;
    }

    virtual const char* name() const {
        static const char* c_name = "Unnamed";
        return c_name;
    }
    
    /*
        \return The selected brush and it's index
    */
    static int selectBrushWithWidget(std::vector<Brush*>& brushes, Options& options, int currentIndex) {
        Brush* selectedBrush = (currentIndex <= brushes.size()) ? brushes.at(currentIndex) : brushes.front();

        if (ImGui::BeginTable("##brushTable", 2)) {
            ImGui::TableSetupColumn("##brushTableColumn1", ImGuiTableColumnFlags_WidthFixed);
            ImGui::TableSetupColumn("##brushTableColumn2", ImGuiTableColumnFlags_WidthStretch);

            ImGui::TableNextColumn();

            ImGui::Text("Brush Type: "); ImGui::TableNextColumn();
            if (ImGui::BeginCombo("##brushTypeCombo", brushes.at(currentIndex)->name()))
            {
                for (int n = 0; n < brushes.size(); n++)
                {
                    if (ImGui::Selectable(brushes.at(n)->name(), currentIndex == n))
                        currentIndex = n;

                    if (currentIndex == n)
                        ImGui::SetItemDefaultFocus();
                }
                ImGui::EndCombo();
            }
            ImGui::TableNextColumn();

            ImGui::Text("Size: "); ImGui::TableNextColumn();
            ImGui::DragInt("##brushSize", &options.brushSize, 1, 1, 64);
            ImGui::TableNextColumn();

            selectedBrush = brushes.at(currentIndex);
            selectedBrush->setRadius(options.brushSize);
            selectedBrush->showImGuiWidget(options);

            ImGui::EndTable();
        }

        return currentIndex;
    }

protected:
	int         m_radius = 5;
    Cell::Type  m_type = Cell::Type::BEDROCK;
    Shape       m_shape = Shape::Circle;

    virtual Cell getCell(const CellCoord& coord) {
        return Cell();
    }

    static void showImGuiCellTypeWidget(Options& options) {
        ImGui::Text("Cell: "); ImGui::TableNextColumn();
        if (ImGui::BeginCombo("##brushCellTypeCombo", Cell::CellUniformProperties::getName(Cell::Type(options.brushCellType))))
        {
            for (int n = 0; n < Cell::typeCount(); n++)
            {
                const char* name = Cell::CellUniformProperties::getName(Cell::Type(n));
                if (strcmp(name, "-") == 0)
                    continue;

                const bool is_selected = (options.brushCellType == n);
                if (ImGui::Selectable(name, is_selected))
                    options.brushCellType = n;

                if (is_selected)
                    ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
        ImGui::TableNextColumn();
    }

};

template <string_literal _Name>
class NamedBrush : public Brush {
public:
    NamedBrush() = default;

    NamedBrush(Cell::Type type)
        : Brush(type)
    { }

    virtual const char* name() const override {
        return _Name.value;
    }
};

using Eraser = NamedBrush<"Eraser">;

class SolidBrush : public NamedBrush<"Solid"> {
public:
    SolidBrush(Cell::Type type, unsigned int shade)
        : NamedBrush(type)
        , m_shade(shade)
    { }

    virtual Brush* showImGuiWidget(Options& options) override {
        showImGuiCellTypeWidget(options);
        showImGuiShadeWidget(options);

        return this;
    }

private:
    unsigned int    m_shade;

    virtual Cell getCell(const CellCoord& coord) override {
        return Cell(m_type, m_shade);
    }

    void showImGuiShadeWidget(Options& options) {
        int v = m_shade;
        ImGui::Text("Shade: "); ImGui::TableNextColumn();
        ImGui::DragInt("##SolidBrushShadeDrag", &v, 1, 0, 64);
        ImGui::TableNextColumn();
        m_shade = v;
    }

};

class RandomizedBrush : public NamedBrush<"Randomizer"> {
public:
    RandomizedBrush(Cell::Type type) { 
        m_type = type;
    }

    virtual Brush* showImGuiWidget(Options& options) override {
        showImGuiCellTypeWidget(options);

        return this;
    }

private:
    virtual Cell getCell(const CellCoord& coord) override {
        return Cell(m_type, rand());
    }

};

class TexturedBrush : public NamedBrush<"Textured"> {
public:
    TexturedBrush(Cell::Type type, const BitmapFileResource& texture)
        : m_texture(texture)
    { 
        m_type = type;
    }

    virtual Brush* showImGuiWidget(Options& options) override {
        showImGuiCellTypeWidget(options);
        showImGuiTextureWidget(options);

        return this;
    }

private:
    BitmapFileResource      m_texture;

    using FileMap = std::unordered_map<std::string, std::filesystem::path>;
    std::vector<Texture>    m_widgetTextures{};

    virtual Cell getCell(const CellCoord& coord) override {
        // Reload resource if updated
        if (m_texture.updated())
            m_texture.load();

        // Read color from texture
        IO::RGB color;
        unsigned x = coord.x % m_texture.value().width(),
                 y = coord.y % m_texture.value().height();
        *((unsigned int*)&color) = m_texture.value().getRGBA(x, y);

        // Decode color to shade
        int shade = floor(color.r / 85.f) + 4 * floor(color.g / 85.f) + 16 * floor(color.b / 85.f);

        return Cell(m_type, shade);
    }

    bool compareSuffix(std::string filename, std::string suffix) {
        return filename.length() >= suffix.length()
            && filename.substr(filename.length() - suffix.length()) == suffix;
    }

    std::pair<FileMap, FileMap> getFilesInDirectory(std::string path) {
        namespace fs = std::filesystem;

        auto res = std::make_pair(FileMap(), FileMap());

        // Get references to the result maps
        auto& [textureFiles, texturePreviewFiles] = res;

        // Iterate over files in the directory with the provided path
        for (const auto& entry : fs::directory_iterator(path)) {
            fs::path path = entry.path();

            // Get filename and declare suffixes to look for
            const std::string filename = path.filename().string();
            const std::string previewSuffix = "_preview.bmp";
            const std::string bmpSuffix = ".bmp";

            // If the file ends with _preview.bmp it is a preview file
            if (compareSuffix(filename, previewSuffix))
                texturePreviewFiles.insert({ filename.substr(0, filename.length() - previewSuffix.length()), path });

            // If the file ends with .bmp it is a texture file
            else if (compareSuffix(filename, bmpSuffix))
                textureFiles.insert({ filename.substr(0, filename.length() - bmpSuffix.length()), path });
        }

        return res;
    }

    void showImGuiTextureWidget(Options& options) {
        namespace fs = std::filesystem;

        ImGui::Text("Texture: "); ImGui::TableNextColumn();

        // Destroy textures created in the previous frame
        m_widgetTextures.clear();

        // Load texture and preview files from data/textures/
        auto [textureFiles, texturePreviewFiles] = getFilesInDirectory("data/textures/");

        int textureCount = textureFiles.size();
        int wrapSize = ImGui::GetWindowPos().x + ImGui::GetWindowContentRegionMax().x;
        int i = 0;

        // Iterate over the texture files
        for (auto [filename, path] : textureFiles) {
            fs::path previewPath;
            std::string pathString = path.string();
        
            // Get the preview file for the texture if it exists
            if (texturePreviewFiles.find(filename) != texturePreviewFiles.end())
                previewPath = texturePreviewFiles.at(filename);

            // Load preview texture or if it doesn't exist load the normal texture
            std::string displayPath = (!previewPath.empty()) 
                ? previewPath.string() : path.string();
            m_widgetTextures.push_back(Texture::loadFromFile(displayPath.c_str(), GL_NEAREST, GL_NEAREST));

            // If texture couldn't be loaded continue
            if (m_widgetTextures.back().empty())
                continue;

            // Check if the currently displayed texture if selected
            bool selected = 0 == options.brushTexturePath.compare(pathString);
            if (selected)
                ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)ImColor::HSV(.5f, .5f, .7f));
            
            // Display texture button
            void* id = (void*)(intptr_t)m_widgetTextures.back().id();
            ImVec2 buttonSize(32, 32);
            if (ImGui::ImageButton((void*)(intptr_t)id, buttonSize)) {

                // Set selected texture
                options.brushTexturePath = pathString;
                m_texture.swap(pathString);
            }

            // Same line with wrapping
            ImGuiStyle& style = ImGui::GetStyle();
            float lastButtonX2 = ImGui::GetItemRectMax().x;
            float nextButtonX2 = lastButtonX2 + style.ItemSpacing.x + buttonSize.x;
            if (i + 1 < textureCount && nextButtonX2 < wrapSize)
                ImGui::SameLine();
            
            if (selected)
                ImGui::PopStyleColor(1);

            // Display filename when the button is hovered
            ImGui::SetItemTooltip(filename.c_str());

            ++i;
        }

        ImGui::TableNextColumn();
    }

};
