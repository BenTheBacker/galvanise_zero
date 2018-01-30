ggp-zero
========

recent
------

[reversi training](https://github.com/ggplib/ggp-zero/blob/dev/doc/reversi_record.md).

[Watch live! *Be back soon*!](http://simulated.tech:8800/index.html/)

Post to https://github.com/mokemokechicken/reversi-alpha-zero/issues/40

about
------

[General Game Playing](https://en.wikipedia.org/wiki/General_game_playing) with
reinforcement learning, starting from zero.

Current games trained:

 * reversi (strong)
 * breakthrough (strong)
 * cittaceot (solved, network reports 99% probability of first move win)
 * hex
 * connect four
 * checkers-variant
 * escort latched breakthrough (variant) and speedChess (variant)

The trained model for reversi is very strong, and plays at ntest level 7.

Based on [GGPLib](https://github.com/ggplib/ggplib).


roadmap
-------
 * flesh out "zero battleground", visualisation and pretty elo graphs

 * train new variant of escort latched breakthrough - which is super hard for mcts players (and
   since method of learning is similar to mcts, can it learn to play?)

 * add previous states & multiple policy heads

 * lots of refactoring

 * update install instructions.  finish refactoring & quick polish up.  write a little about how it works.  post first working version.


other
-----
* old / early [results](https://github.com/ggplib/ggp-zero/blob/dev/doc/old_results.md).
* old install [instructions](https://github.com/ggplib/ggp-zero/blob/dev/doc/install.md).
