#pragma once
#include "../utils.h"

#define _CELL_BITS_T short int
#define _CELL_PROP_HEADER _CELL_BITS_T : s_headerSizeInBits

#define _MAX_TYPE_COUNT ~((unsigned char)-1 << s_typeBitCount)

// In order to improve Cell size change _CELL_BITS_T to find best alignment. 
class Cell {
private:
	using EnumBase = unsigned _CELL_BITS_T;

public:
	enum class Form : EnumBase {
		GAS = 0b00,
		LIQUID = 0b01,
		SOLID = 0b10,
	};

private:
	constexpr static EnumBase s_typeBitCount = 5;
	constexpr static EnumBase s_typeFormBitCounts[2] = { 2, 1 };

	template <Form _Form>
	constexpr static EnumBase s_leftShiftForm_v = ((EnumBase)_Form << s_typeBitCount - 2);

	constexpr static unsigned int s_velocityBitCount = 3;
	constexpr static unsigned int s_headerSizeInBits = 1 + s_typeBitCount + s_velocityBitCount * 2;

public:
	enum class Type : EnumBase {
		AIR = s_leftShiftForm_v<Form::GAS> | 0b000,

		WATER = s_leftShiftForm_v<Form::LIQUID> | 0b000,

		BEDROCK = s_leftShiftForm_v<Form::SOLID> | 0b0000,
		STONE = s_leftShiftForm_v<Form::SOLID> | 0b0001,
		DIRT = s_leftShiftForm_v<Form::SOLID> | 0b0010,
		SAND = s_leftShiftForm_v<Form::SOLID> | 0b0011,
	};

	struct SandProperties {
		_CELL_PROP_HEADER;

	};

	struct WaterProperties {
		_CELL_PROP_HEADER;

		unsigned _CELL_BITS_T direction : 1;
	};

	union {
		struct {
			unsigned _CELL_BITS_T updatedOnEvenTick : 1;	// Indicates weather the cell was last updated on an even number tick.
			Type type : s_typeBitCount;							// The type and form of the cell.
			_CELL_BITS_T velocityX : s_velocityBitCount;
			_CELL_BITS_T velocityY : s_velocityBitCount;
			unsigned _CELL_BITS_T shade : 6;
		};

		SandProperties sand;
		WaterProperties water;
	};

	__host__ __device__
	Cell() {
		memset(this, 0, sizeof(*this));
	}

	Cell(const Json& json);

	__device__ __host__
	static Cell create(Type type);

	Form form() const {
		return (Form)(((EnumBase)type & formMask(type)) >> (s_typeBitCount - 2));
	}

	bool isLighter(Cell cell) const {
		return cell.form() > form();
	}

	__device__ __host__
	constexpr static unsigned char MaxTypeCount() {
		return ~((unsigned char)-1 << s_typeBitCount);
	}

private:
	constexpr static EnumBase formMask(Type type) {
		return (EnumBase)-1 << s_typeBitCount - s_typeFormBitCounts[(EnumBase)type >> (s_typeBitCount - 1)];
	}
};

template <>
inline Json& Json::operator=(Cell chunk) {
	*this = Json::CreateEmptyArray();
	for (int i = 0; i < sizeof(Cell); i++)
		this->array.push_back(Json(((unsigned char*)&chunk)[i]));
	return *this;
}

inline Cell::Cell(const Json& json) {
	for (int i = 0; i < sizeof(Cell); i++)
		((unsigned char*)this)[i] = json.array.at(i);
}

namespace SimValues {

__host__ __device__
inline Cell Air(bool updatedOnEvenTick = false) {
	Cell res; res.type = Cell::Type::AIR; res.updatedOnEvenTick = updatedOnEvenTick; res.shade = rand(); return res;
}

__host__ __device__
inline Cell Water(bool updatedOnEvenTick = false) {
	Cell res; res.type = Cell::Type::WATER; res.updatedOnEvenTick = updatedOnEvenTick; res.shade = rand(); return res;
}

__host__ __device__
inline Cell Sand(bool updatedOnEvenTick = false) {
	Cell res; res.type = Cell::Type::SAND; res.updatedOnEvenTick = updatedOnEvenTick; res.shade = rand(); return res;
}

__host__ __device__
inline Cell Bedrock(bool updatedOnEvenTick = false) {
	Cell res; res.type = Cell::Type::SAND; res.updatedOnEvenTick = updatedOnEvenTick; res.shade = rand(); return res;
}

}

inline Cell Cell::create(Type type) {
	Cell res; res.type = type; res.shade = rand();

	switch (type)
	{
	case Cell::Type::SAND: return SimValues::Sand(); break;
	default:
		return res;
		break;
	}
}
