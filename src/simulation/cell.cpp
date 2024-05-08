#include "cell.h"
#include "chunk.h"

void updateGold(Cell& cell, Chunk& chunk, int x, int y);
void updateDirt(Cell& cell, Chunk& chunk, int x, int y);
void updateVegetation(Cell& cell, Chunk& chunk, int x, int y);

void Cell::update(Cell& cell, Chunk& chunk, int x, int y) {
	// Return if cell was allready updated in this tick
	if (chunk.isCellUpdated(cell))
		return;

	switch (cell.type)
	{
	case Cell::Type::AIR:
		break;
	case Cell::Type::WATER:
		break;
	case Cell::Type::BEDROCK:
		break;
	case Cell::Type::STONE:
		movePowder(cell, chunk, x, y);
		break;
	case Cell::Type::DIRT:
		movePowder(cell, chunk, x, y);
		updateDirt(cell, chunk, x, y);
		break;
	case Cell::Type::SAND:
		movePowder(cell, chunk, x, y);
		break;
	case Cell::Type::GOLD:
		movePowder(cell, chunk, x, y);
		updateGold(cell, chunk, x, y);
		break;
	case Cell::Type::VEGETATION:
		updateVegetation(cell, chunk, x, y);
		break;
	default:
		break;
	}
}

void Cell::movePowder(Cell& cell, Chunk& chunk, int x, int y) {
	CellCoord destination(x, y - 1);

	if (chunk.getCell(destination).isLighter(cell)) {
		cell.updatedOnEvenTick = chunk.evenTick();
		cell.isNotFreeFalling = false;

		chunk.cell({ x + 1, y }).isNotFreeFalling = false;
		chunk.cell({ x - 1, y }).isNotFreeFalling = false;

		chunk.swapCells({ x, y }, destination);
		return;
	}

	if (cell.isNotFreeFalling)
		return;

	char displacement = chunk.evenTick() * 2 - 1;
	destination.x += displacement;
	if (chunk.getCell(destination).isLighter(cell)) {
		chunk.swapCells({ x, y }, destination);
		chunk.markCellAsUpdated(destination);

		chunk.cell({ destination.x + 1, destination.y + 1 }).isNotFreeFalling = false;
		chunk.cell({ destination.x - 1, destination.y + 1 }).isNotFreeFalling = false;

		return;
	}

	destination.x -= 2 * displacement;
	if (chunk.getCell(destination).isLighter(cell)) {
		chunk.swapCells({ x, y }, destination);
		chunk.markCellAsUpdated(destination);

		chunk.cell({ destination.x + 1, destination.y + 1 }).isNotFreeFalling = false;
		chunk.cell({ destination.x - 1, destination.y + 1 }).isNotFreeFalling = false;

		return;
	}

	if (cell.isFreeFalling())
		cell.isNotFreeFalling = true;
}

void Cell::moveLiquid(Cell cell, Chunk& chunk, int x, int y) {

}

Cell::CellUniformProperties::CellUniformProperties()
{
	for (int i = 0; i < Cell::typeCount(); ++i) {
		types[i] = Cell::Type(i);
		m_names[i] = "-";
	}

	m_names[(int)Cell::Type::AIR]		= "Air";
	m_names[(int)Cell::Type::WATER]		= "Water";
	m_names[(int)Cell::Type::BEDROCK]	= "Bedrock";
	m_names[(int)Cell::Type::STONE]		= "Stone";
	m_names[(int)Cell::Type::DIRT]		= "Dirt";
	m_names[(int)Cell::Type::SAND]      = "Sand";
	m_names[(int)Cell::Type::GOLD]		= "Gold";
	m_names[(int)Cell::Type::VEGETATION]= "Vegetation";
}

Cell::CellUniformProperties& Cell::CellUniformProperties::instance()
{
	CellUniformProperties c_instance;
	return c_instance;
}

void updateDirt(Cell& cell, Chunk& chunk, int x, int y) {
	if (cell.dirt.exhausted || cell.isFreeFalling() || chunk.getCell({ x, y + 1 }).type != Cell::Type::AIR)
		return;
	
	if (rand() % 100 == 0) {
		if (cell.dirt.exhausted = rand() % 3 == 0; cell.dirt.exhausted)
			return;

		chunk.setCell({ x, y + rand() % 2 }, Cell(Cell::Type::VEGETATION, 0));
	}

	chunk.setCell({ x, y }, cell);
}

void updateVegetation(Cell& cell, Chunk& chunk, int x, int y) {
	if (chunk.getCell({ x, y + 1 }).type != Cell::Type::AIR)
		return;

	if (rand() % 1000 == 0)
		chunk.setCell({ x, y + 1 }, Cell(Cell::Type::VEGETATION, 1));

	chunk.setCell({ x, y }, cell);
}

void updateGold(Cell& cell, Chunk& chunk, int x, int y) {
	if (cell.gold.exhausted)
		return;

	int exhaustProbabilityInverse = 3;

	if (rand() % 10 == 0)
	{
		if (chunk.getCell({ x, y + 1 }).type == Cell::Type::BEDROCK) {
			if (chunk.getCell({ x - 1, y + 1 }).type == Cell::Type::BEDROCK) if (chunk.getCell({ x - 2, y + 1 }).type == Cell::Type::BEDROCK)
			if (chunk.getCell({ x + 1, y + 1 }).type == Cell::Type::BEDROCK) if (chunk.getCell({ x + 2, y + 1 }).type == Cell::Type::BEDROCK)
			if (chunk.getCell({ x, y + 2 }).type == Cell::Type::BEDROCK)
			if (chunk.getCell({ x - 1, y + 2 }).type == Cell::Type::BEDROCK) if (chunk.getCell({ x - 2, y + 2 }).type == Cell::Type::BEDROCK)
			if (chunk.getCell({ x + 1, y + 2 }).type == Cell::Type::BEDROCK) if (chunk.getCell({ x + 2, y + 2 }).type == Cell::Type::BEDROCK)
			if (chunk.getCell({ x, y + 3 }).type == Cell::Type::BEDROCK)
			if (chunk.getCell({ x - 1, y + 3 }).type == Cell::Type::BEDROCK) if (chunk.getCell({ x - 2, y + 3 }).type == Cell::Type::BEDROCK)
			if (chunk.getCell({ x + 1, y + 3 }).type == Cell::Type::BEDROCK) if (chunk.getCell({ x + 2, y + 3 }).type == Cell::Type::BEDROCK)

			{
				Cell res = Cell(Cell::Type::GOLD, rand() % 5);
				res.gold.exhausted = rand() % exhaustProbabilityInverse == 0;
				chunk.setCell({ x, y + 1 }, res);
			}
		}

		if (chunk.getCell({ x + 1, y }).type == Cell::Type::BEDROCK) {
			if (chunk.getCell({ x + 1, y - 1 }).type == Cell::Type::BEDROCK) if (chunk.getCell({ x + 1, y - 2 }).type == Cell::Type::BEDROCK)
			if (chunk.getCell({ x + 1, y + 1 }).type == Cell::Type::BEDROCK) if (chunk.getCell({ x + 1, y + 2 }).type == Cell::Type::BEDROCK)
			if (chunk.getCell({ x + 2, y }).type == Cell::Type::BEDROCK)
			if (chunk.getCell({ x + 2, y - 1 }).type == Cell::Type::BEDROCK) if (chunk.getCell({ x + 2, y - 2 }).type == Cell::Type::BEDROCK)
			if (chunk.getCell({ x + 2, y + 1 }).type == Cell::Type::BEDROCK) if (chunk.getCell({ x + 2, y + 2 }).type == Cell::Type::BEDROCK)
			if (chunk.getCell({ x + 3, y }).type == Cell::Type::BEDROCK)
			if (chunk.getCell({ x + 3, y - 1 }).type == Cell::Type::BEDROCK) if (chunk.getCell({ x + 3, y - 2 }).type == Cell::Type::BEDROCK)
			if (chunk.getCell({ x + 3, y + 1 }).type == Cell::Type::BEDROCK) if (chunk.getCell({ x + 3, y + 2 }).type == Cell::Type::BEDROCK)

			{
				Cell res = Cell(Cell::Type::GOLD, rand() % 5);
				res.gold.exhausted = rand() % exhaustProbabilityInverse == 0;
				chunk.setCell({ x + 1, y }, res);
			}
		}

		if (chunk.getCell({ x, y - 1 }).type == Cell::Type::BEDROCK) {
			if (chunk.getCell({ x - 1, y - 1 }).type == Cell::Type::BEDROCK) if (chunk.getCell({ x - 2, y - 1 }).type == Cell::Type::BEDROCK)
			if (chunk.getCell({ x + 1, y - 1 }).type == Cell::Type::BEDROCK) if (chunk.getCell({ x + 2, y - 1 }).type == Cell::Type::BEDROCK)
			if (chunk.getCell({ x, y - 2 }).type == Cell::Type::BEDROCK)
			if (chunk.getCell({ x - 1, y - 2 }).type == Cell::Type::BEDROCK) if (chunk.getCell({ x - 2, y - 2 }).type == Cell::Type::BEDROCK)
			if (chunk.getCell({ x + 1, y - 2 }).type == Cell::Type::BEDROCK) if (chunk.getCell({ x + 2, y - 2 }).type == Cell::Type::BEDROCK)
			if (chunk.getCell({ x, y - 3 }).type == Cell::Type::BEDROCK)
			if (chunk.getCell({ x - 1, y - 3 }).type == Cell::Type::BEDROCK) if (chunk.getCell({ x - 2, y - 3 }).type == Cell::Type::BEDROCK)
			if (chunk.getCell({ x + 1, y - 3 }).type == Cell::Type::BEDROCK) if (chunk.getCell({ x + 2, y - 3 }).type == Cell::Type::BEDROCK)

			{
				Cell res = Cell(Cell::Type::GOLD, rand() % 5);
				res.gold.exhausted = rand() % exhaustProbabilityInverse == 0;
				chunk.setCell({ x, y - 1 }, res);
			}
		}

		if (chunk.getCell({ x - 1, y }).type == Cell::Type::BEDROCK) {
			if (chunk.getCell({ x - 1, y + 1 }).type == Cell::Type::BEDROCK) if (chunk.getCell({ x - 1, y + 2 }).type == Cell::Type::BEDROCK)
			if (chunk.getCell({ x - 1, y - 1 }).type == Cell::Type::BEDROCK) if (chunk.getCell({ x - 1, y - 2 }).type == Cell::Type::BEDROCK)
			if (chunk.getCell({ x - 2, y }).type == Cell::Type::BEDROCK)
			if (chunk.getCell({ x - 2, y + 1 }).type == Cell::Type::BEDROCK) if (chunk.getCell({ x - 2, y + 2 }).type == Cell::Type::BEDROCK)
			if (chunk.getCell({ x - 2, y - 1 }).type == Cell::Type::BEDROCK) if (chunk.getCell({ x - 2, y - 2 }).type == Cell::Type::BEDROCK)
			if (chunk.getCell({ x - 3, y }).type == Cell::Type::BEDROCK)
			if (chunk.getCell({ x - 3, y + 1 }).type == Cell::Type::BEDROCK) if (chunk.getCell({ x - 3, y + 2 }).type == Cell::Type::BEDROCK)
			if (chunk.getCell({ x - 3, y - 1 }).type == Cell::Type::BEDROCK) if (chunk.getCell({ x - 3, y - 2 }).type == Cell::Type::BEDROCK)

			{
				Cell res = Cell(Cell::Type::GOLD, rand() % 5);
				res.gold.exhausted = rand() % exhaustProbabilityInverse == 0;
				chunk.setCell({ x - 1, y }, res);
			}
		}

	}
	chunk.setCell({ x, y }, cell);
}
