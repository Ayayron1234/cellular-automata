#pragma once
#include "mutex"
#include "common.h"
#include "cell.h"
#include "../utils.h"

class ChunkState {
private:
	enum class UpdateState { Unchanged = 0, Updated };

public:
	void drawn() {
		std::lock_guard<std::mutex> lock(m_mut);

		m_updateSinceDraw = UpdateState::Unchanged;
	}

	// Called when cell was set
	void cellSet(Cell::Type previousType, Cell::Type currentType) {
		std::lock_guard<std::mutex> lock(m_mut);

		// Update count of cells in chunk which are not air
		//		aa -> +1 -1
		//		an -> +1  0
		//		na ->  0 -1 
		//		nn ->  0  0
		m_nonAirCount += previousType == Cell::Type::AIR;
		m_nonAirCount -= currentType == Cell::Type::AIR;

		m_updateState = UpdateState::Updated;
		m_updateSinceDraw = UpdateState::Updated;
	}

	void requestUpdate() {
		std::lock_guard<std::mutex> lock(m_mut);

		m_updateState = UpdateState::Updated;
		m_updateSinceDraw = UpdateState::Updated;
	}

	// Called before chunk is processed
	void nextState() {
		std::lock_guard<std::mutex> lock(m_mut);

		m_shouldUpdate = (bool)m_updateState;
		m_updateState = UpdateState::Unchanged;
	}

	bool updatedSinceDraw() const {
		return m_updateSinceDraw == UpdateState::Updated;
	}

	bool shouldUpdate() {
		std::lock_guard<std::mutex> lock(m_mut);

		return m_shouldUpdate;
	}

	bool empty() const {
		return m_nonAirCount == 0;
	}

	void deserialise(std::ifstream& is) {
		is.read((char*)&m_nonAirCount, sizeof(m_nonAirCount));
	}

	void serialise(std::ofstream& os) const {
		os.write((char*)&m_nonAirCount, sizeof(m_nonAirCount));
	}

private:
	UpdateState m_updateState = UpdateState::Updated;
	unsigned	m_nonAirCount = 0;
	bool		m_shouldUpdate;

	UpdateState m_updateSinceDraw = UpdateState::Updated;

	std::mutex  m_mut;

};
