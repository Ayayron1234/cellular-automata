#pragma once

/*
Teljesítménnyel kapcsolatos kérdés: 

ha az algoritmusomhoz ezt a struct-ot használom:

struct Cell {
    bool updatedOnEven: 1;
    unsigned int type: 7;
}

akkor kb fele annyi ideig tart az algoritmus végrehajtása (1 szálon, teszthez 1.048.576 cell-t használtam), mint ha ezt használom ugyan azzal az algoritmussal:

struct Cell {
    bool updatedOnEven: 1;
    unsigned int type: 6;
    bool direction: 1;
}

Első gondolatom az volt, hogy kevésbbé hatékony cache-elés miatt van, de mindkét struct ugyan akkora helyet fogla...

Oh geci mégis cache. 


*/




struct Cell {
    short unsigned int updatedOnEven: 1;
    short unsigned int type: 6;
    short unsigned int waterDirection: 1;
    short int velocityX : 5;
    short int velocityY : 5;
    //bool waterDirection : 1;
};
