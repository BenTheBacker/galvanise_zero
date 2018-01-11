#pragma once

#include "supervisorbase.h"
#include "puct/evaluator.h"

#include <statemachine/statemachine.h>

namespace GGPZero {
    // forwards
    class PuctEvaluator;

    class TestSelfPlay {
    public:
        TestSelfPlay(SupervisorBase* supervisor, const GGPLib::BaseState* initial_state);
        ~TestSelfPlay();

    public:
        void playOnce();

    private:
        SupervisorBase* supervisor;
        PuctEvaluator pe;
        const GGPLib::BaseState* initial_state;
    };

}
