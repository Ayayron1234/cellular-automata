#pragma once

struct Cell {
    bool updatedOnEven : 1;
    unsigned int type : 7;
};
