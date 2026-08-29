"""Records a policy playing, straight to an animated GIF, with no window and no video step.

    cd snek3
    PYTHONPATH=. python -u record_gif.py hof                       # the HOF record, 60s
    PYTHONPATH=. python -u record_gif.py hof --seconds 30 --fps 25
    PYTHONPATH=. python -u record_gif.py hallOfFame/<entry> --seed 7 --tile 40
    PYTHONPATH=. python -u record_gif.py --list                    # what is in hallOfFame/

The obvious way to build this would be to open a window, screen-record it, cut sixty seconds out
of the recording and transcode that to a GIF. Every one of those steps is avoidable. `Game` draws
into a pygame surface and only *flips* that surface to the window server, so under
`SDL_VIDEODRIVER=dummy` the drawing all still happens and the flip becomes a no-op — the frames
are already in memory, one per game step, and can be read out with `pygame.image.tostring`.
Measured at **0.68 ms/frame**, so a whole episode is captured in under a second on one core, with
no window, no screen recorder, no ffmpeg (which is not installed on either host) and no
transcode. Nothing here can affect a training run: it only reads checkpoint files.

**The frame rate is not a free parameter, and asking for 60 fps gets you 10.** A GIF stores each
frame's delay in *hundredths* of a second, so the only representable rates are 100, 50, 33.3, 25
and 20 fps; 60 fps rounds to a 10 ms delay, and browsers clamp any delay of 10 ms or less to
**100 ms**, i.e. 10 fps — slower than the 20 fps version and in the wrong direction. So
`gif_delay_ms` quantises to the 10 ms grid and refuses to go below 20 ms, making **50 fps the
ceiling**. `--format webp` has no such grid if an exact rate is ever wanted.

**One frame is one game step, so the rate and the length are coupled.** A perfect game from a
champion checkpoint is 1200-1600 steps, which is only 24-32 s at 50 fps.

**A recording is a whole number of complete games, and `--seconds` is a floor rather than a
target**: it runs for `--min-games` games *or* `--seconds`, whichever is longer. Three games from a
champion is 80-95 s, so the default 60 s floor is usually not the binding one; a weaker policy whose
games are short keeps playing until the clock is covered. Nothing is ever cut off mid-game to fit a
duration, and nothing is time-lapsed to fit one — a small shortfall under the floor is absorbed by
holding the final frame, which is free.

Two things the game's own rendering does that a naive capture inherits, both fixed here rather
than in `env/game.py`:

- **The winning frame is blank.** `Game.step` erases the sprites, then `render()` sees
  `perfect_game` and returns *before* redrawing them, so it blits "PERFECT GAME!!!" onto an empty
  board. The filled board — the whole point of the recording — is never drawn by the game at all.
  `redraw_board` draws it, and the message goes on top of it afterwards.
- **The HUD is drawn under the sprites**, so the score and step readouts are buried as the board
  fills and are illegible for most of a winning game. `suppress_game_hud` paints them in the
  background colour and a real header is composited outside the board instead.

`random.seed` fully determines the game: food placement is the only randomness in `env/game.py` and a
greedy policy is deterministic. So `--seed` reproduces a recording exactly, and the seed of every
game is printed for that reason.

Environment: none of its own. `SNEK_TILE_PIXELS` is set from `--tile` before `env.render` is
imported, because every pixel constant is derived from it at import time.
"""
import os
import sys

# Before any pygame or `env.render` import. The dummy driver is what makes this headless *and* what
# makes it fast; audio is silenced for the reason every entry point does it.
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'

import argparse
import random
import time


# `--tile` has to reach `env.render` before it is imported, so argv is read here rather than in
# main(). Everything else is parsed normally below.
def _preparse_tile(argv):
    for i, arg in enumerate(argv):
        if arg == '--tile' and i + 1 < len(argv):
            return arg, argv[i + 1]
        if arg.startswith('--tile='):
            return arg, arg.split('=', 1)[1]
    return None, None


_flag, _tile = _preparse_tile(sys.argv[1:])
os.environ['SNEK_TILE_PIXELS'] = _tile if _tile else '30'

import pygame
from PIL import Image, ImageDraw, ImageFont

from env import constants
from env import render as R
from env.constants import GIFS_DIR, HOF_DIR, POLICY_DIR
from tools import checkpoints

# The current record, so `record_gif.py hof` needs no arguments. **`hallOfFame/HOF.md` is the
# authority on which entry that is** — this is a convenience shortcut, not a second source of
# truth, and it is one line to move. None until snek3 has a record of its own, in which case `hof`
# means "the only entry" and says so if there is more than one.
HOF_RECORD = None

PERFECT_TEXT = 'PERFECT GAME!!!'

# GIF delays live on a 10 ms grid, and a delay at or below 10 ms is clamped to 100 ms by every
# major browser. Both numbers are load-bearing; see the module docstring.
GIF_DELAY_UNIT_MS = 10
MIN_SAFE_DELAY_MS = 20
# 255 rather than 256: Pillow reserves an index for the transparency slot of an optimised
# animation, and asking for 256 costs a colour rather than gaining one.
MAX_GIF_COLORS = 255

HEADER_BG = (255, 255, 255)
HEADER_RULE = (219, 219, 222)
TITLE_COLOR = (24, 24, 26)
SUBTITLE_COLOR = (110, 110, 116)

# Held beats, in frames at the recording's own rate, converted from seconds so they read the same
# at any --fps. The finished board is shown before the message so the completed board is actually
# seen, which is the frame the whole recording exists for.
HOLD_OPEN_S = 0.4
HOLD_BOARD_S = 0.5
HOLD_MESSAGE_S = 1.4
# A death gets a shorter beat than a win: it is punctuation between games rather than the payoff,
# and with --allow-losses there can be several of them. Long enough to register either way — a
# death frame with no hold at all is a single 20 ms frame, which is invisible.
HOLD_DEATH_S = 0.5

# How much of the requested length may be filled by holding the final frame, rather than by
# playing one more game. This is what stops a 34-frame shortfall from costing a whole extra game:
# a third 1500-frame game to cover 34 frames forces a 35% resample, and dropping one frame in
# three is visible as skipped motion, while two extra seconds on a *static* winning board is not
# visible at all. So the recorder stops as soon as it is within this of the target and pads.
PAD_BUDGET_S = 2.0


# ----------------------------- timing, pure and tested ----------------------------- #

def gif_delay_ms(fps):
    """The per-frame delay a GIF can actually carry for `fps`, in ms.

    Quantised to the 10 ms grid the container stores, and floored at 20 ms: a 10 ms delay is
    clamped to 100 ms by browsers, so 100 fps is not merely unattainable, it plays at 10 fps.
    """
    delay = int(round((1000.0 / float(fps)) / GIF_DELAY_UNIT_MS)) * GIF_DELAY_UNIT_MS
    return max(MIN_SAFE_DELAY_MS, delay)


def target_frame_count(seconds, delay_ms):
    """Frames needed for `seconds` of playback at `delay_ms` per frame."""
    return max(1, int(round(seconds * 1000.0 / delay_ms)))


def resample(frames, target):
    """`frames` stretched or squeezed to exactly `target` entries, keeping the first and last.

    Used to sample a spread of frames for the shared palette. It is deliberately *not* used to fit
    the duration: a recording is a whole number of games at one frame per game step, so fitting is
    done by choosing how many games to play, not by thinning them.
    """
    count = len(frames)
    if count == target:
        return list(frames)
    if target == 1:
        return [frames[-1]]
    last = count - 1
    return [frames[min(last, int(round(i * last / float(target - 1))))] for i in range(target)]


def fit_length(captured, target, pad_budget):
    """`(final_length, pad)` for `captured` frames against a `target` floor.

    This is "`--min-games` games or `--seconds`, whichever is longer", in one place:

    - **Longer than the floor: keep everything.** The games are already complete, so the recording
      simply runs long. Thinning it to hit the clock would turn a recording of a game into a
      time-lapse of one, which is why nothing here ever drops a frame.
    - **Short of the floor by at most `pad_budget`: pad.** Holding the final frame is free in bytes
      and invisible on a board that is already static, where playing a whole extra game to cover a
      few frames is neither.
    - **Short by more than that: run short.** Only reachable when the attempt budget ran out, and
      the caller says so — a freeze of arbitrary length is worse than an honest short recording.
    """
    if captured >= target:
        return captured, 0
    shortfall = target - captured
    if shortfall <= pad_budget:
        return target, shortfall
    return captured, 0


# ----------------------------- the two rendering fixes ----------------------------- #

def suppress_game_hud():
    """Paints the in-board HUD in the background colour, i.e. hides it.

    `render()` blits the score, step and policy readouts *before* `self.all.draw()`, so the sprites
    cover them and they are illegible from about half a board on. A header composited outside the
    board replaces them.

    Assigning on `env.render` is all that is needed, and that is the point of splitting the drawing
    constants out: `env.game` reads every one of them as `R.NAME`, so a module-level assignment
    reaches it. snek2's version had to patch `Snake`'s own globals, because its
    `from snake_constants import *` had bound copies at import.
    """
    R.SCORE_COLOR = R.BACKGROUND_COLOR
    R.STEP_COLOR = R.BACKGROUND_COLOR
    R.POLICY_COLOR = R.BACKGROUND_COLOR


def disable_display_throttles():
    """Removes the two places rendering waits on a human.

    `SCORE_SLOW_THRESHOLD` is 248 against a maximum score of 95, so that branch is unreachable
    today — it is neutralised anyway because it would silently drop capture to 10 fps if the score
    scale ever changed. `PERFECT_GAME_WAIT_MS` would otherwise stall every win by half a second.
    """
    R.SCORE_SLOW_THRESHOLD = 10 ** 9
    R.PERFECT_GAME_WAIT_MS = 0


def redraw_board(game):
    """Draws the board that `render()` skips on the winning step.

    `step()` clears the sprites to the background before `render()` runs, and `render()` returns
    early when `perfect_game` is set, so the last frame the game produces is an empty board with a
    message on it. This is the filled board.
    """
    game.screen.blit(game.bg, (0, 0))
    game.all.draw(game.screen)


def overlay_win_message(game):
    """Blits "PERFECT GAME!!!" over whatever is on the surface, without erasing it.

    Same font sizing the game uses — `_message_font` shrinks until the text fits the window — so
    the message looks like the game's own, just over a full board instead of a blank one.
    """
    font = game._message_font(PERFECT_TEXT, R.PERFECT_FONT_SIZE)
    image = font.render(PERFECT_TEXT, True, (0, 0, 0))
    game.screen.blit(image, image.get_rect(center=R.SCREENRECT.center))


def overlay_death_message(game):
    """Blits the game's own death text over the board, without erasing it.

    `render()` picks between 'NO FUD' and 'DED' by `starved`, at two different font sizes, and this
    mirrors that rather than inventing a third message — the recording should look like the game.
    The same early-return that blanks the winning frame blanks this one, so it needs the same
    `redraw_board` first.
    """
    if game.starved:
        text, size = 'NO FUD', R.STARVE_FONT_SIZE
    else:
        text, size = 'DED', R.DEATH_FONT_SIZE
    font = game._message_font(text, size)
    image = font.render(text, True, (0, 0, 0))
    game.screen.blit(image, image.get_rect(center=R.SCREENRECT.center))


def grab(game):
    """The current surface as a PIL image. One flip's worth of pixels, without the flip."""
    return Image.frombytes('RGB', game.screen.get_size(),
                           pygame.image.tostring(game.screen, 'RGB'))


# ----------------------------- header and encoding ----------------------------- #

def _load_font(size, bold=False):
    """A real font if the host has one, else PIL's builtin. Never raises: this is decoration."""
    candidates = ['/System/Library/Fonts/Helvetica.ttc',
                  '/System/Library/Fonts/Supplemental/Arial.ttf',
                  '/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf' % ('-Bold' if bold else ''),
                  '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf']
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size)
    except Exception:
        return ImageFont.load_default()


class Composer:
    """Pastes a captured board under a header of its own.

    The header exists because the game's HUD cannot be read once the board fills. It is drawn
    outside the board rather than over it, so it never hides a cell.
    """

    def __init__(self, board_size, title, enabled=True, scale=1):
        self.enabled = enabled
        self.scale = scale
        self.title = title
        self.title_font = _load_font(max(13, int(17 * scale * board_size[0] / 300.0)))
        self.line_font = _load_font(max(11, int(14 * scale * board_size[0] / 300.0)))
        self.height = 0
        if enabled:
            ascent = self.title_font.getbbox('Ag')[3] + self.line_font.getbbox('Ag')[3]
            self.height = int(ascent + 22 * scale)

    def compose(self, board, score, step, game_index, game_count):
        if self.scale != 1:
            board = board.resize((board.width * self.scale, board.height * self.scale),
                                 Image.NEAREST)
        if not self.enabled:
            return board
        canvas = Image.new('RGB', (board.width, board.height + self.height), HEADER_BG)
        canvas.paste(board, (0, self.height))
        draw = ImageDraw.Draw(canvas)
        pad = max(8, int(10 * self.scale))
        draw.text((pad, int(pad * 0.6)), self.title, font=self.title_font, fill=TITLE_COLOR)
        line = 'meals %2d/%d   step %4d' % (score, constants.MAX_POSSIBLE_SCORE, step)
        if game_count > 1:
            line += '   game %d/%d' % (game_index, game_count)
        draw.text((pad, self.height - self.line_font.getbbox('Ag')[3] - int(9 * self.scale)),
                  line, font=self.line_font, fill=SUBTITLE_COLOR)
        draw.line([(0, self.height - 1), (canvas.width, self.height - 1)], fill=HEADER_RULE)
        return canvas


def to_shared_palette(frames, colors=MAX_GIF_COLORS):
    """Every frame quantised to one palette, so the animation cannot flicker between palettes.

    The palette is derived from a strip of frames sampled across the whole recording, not from a
    single frame: the opening board, a mid-game board and the win message do not share a colour
    set, and a palette built from any one of them would dither the others.

    `colors` is worth turning down for a gallery. The board is six flat colours — background,
    body, head, food, title, subtitle — and everything above that is antialiasing on text, so a
    smaller palette costs almost nothing visible and compresses better: measured on a 4334-frame
    recording, 255 colours is 4.35 MB, 64 is 3.55, 16 is 2.88 and 8 is 2.74. It is *not* the main
    lever, though. Size here is driven by frame count, not by palette or by area — a 4x reduction
    in pixels saves only 29%, and animated WebP is three times *worse* (13.5 MB lossless).
    """
    sample = resample(frames, min(8, len(frames)))
    strip = Image.new('RGB', (sample[0].width, sample[0].height * len(sample)))
    for i, frame in enumerate(sample):
        strip.paste(frame, (0, i * frame.height))
    palette = strip.convert('P', palette=Image.ADAPTIVE,
                             colors=max(2, min(MAX_GIF_COLORS, colors)))
    return [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]


def write_animation(frames, out_path, delay_ms, fmt, colors=MAX_GIF_COLORS):
    """Writes the frame list as an animated GIF or WebP.

    Identical consecutive frames are merged by the encoder with their durations summed, so the
    held beats at the start and end cost playback time and almost no bytes.
    """
    if fmt == 'webp':
        frames[0].save(out_path, format='WEBP', save_all=True, append_images=frames[1:],
                       duration=delay_ms, loop=0, lossless=True, method=4)
        return
    quantised = to_shared_palette(frames, colors)
    quantised[0].save(out_path, format='GIF', save_all=True, append_images=quantised[1:],
                      duration=delay_ms, loop=0, optimize=True, disposal=1)


# ----------------------------- checkpoint resolution ----------------------------- #

def hof_entries():
    if not os.path.isdir(HOF_DIR):
        return []
    return sorted(n for n in os.listdir(HOF_DIR)
                  if os.path.isdir(os.path.join(HOF_DIR, n)))


def resolve_policy(spec, step):
    """`(checkpoint_dir, step, label)` for `hof`, a hall-of-fame entry, a path, or a policy name.

    The step is named explicitly rather than taken as "whatever is newest" by a library call, which
    is how snek2's `tf.train.latest_checkpoint` returned None inside a hall-of-fame directory — there
    is no state file there, only the checkpoint — and then failed the restore silently.
    """
    if spec == 'hof':
        entries = hof_entries()
        spec = HOF_RECORD or (entries[0] if len(entries) == 1 else None)
        if spec is None:
            raise SystemExit('hallOfFame/ holds {0} entries, so `hof` is ambiguous. Name one:\n  '
                             '{1}'.format(len(entries), '\n  '.join(entries) or '(none)'))
    for candidate in (spec, os.path.join(HOF_DIR, spec), os.path.join(POLICY_DIR, spec)):
        if os.path.isdir(candidate):
            directory = candidate
            break
    else:
        raise SystemExit('no such policy or checkpoint directory: %s\n'
                         'hallOfFame/ holds:\n  %s' % (spec, '\n  '.join(hof_entries())))

    steps = checkpoints.steps(directory)
    if not steps:
        raise SystemExit('no ckpt-*.pt files in %s' % directory)
    if step is None:
        step = steps[-1]
    elif step not in steps:
        raise SystemExit('no checkpoint for step %d in %s (have %s)'
                         % (step, directory, ', '.join(str(s) for s in steps)))

    name = os.path.basename(os.path.normpath(directory))
    # 'b44a-lowlr7-b29b-ckpt2739000' names its own step; don't say it twice.
    short = name.split('-ckpt')[0]
    label = '%s @%dk' % (short, step // 1000)
    return directory, step, label


# ----------------------------- playing and capturing ----------------------------- #

def play_episode(env, policy_fn):
    """One greedy episode, captured. Returns `(boards, meta, won, info)`.

    `boards` holds one frame per game step plus the opening frame; `meta` the `(score, step)` for
    each. On a win the last board is replaced with the filled board the game does not draw, and a
    second copy carrying the message is appended.
    """
    game = env.game
    observation = env.reset()
    # `reset()` builds the sprites but does not draw them — the game only draws inside `render()`,
    # which nothing calls until the first step. Without this the opening frame of every recording is
    # a blank board, which snek2's recordings all had.
    game.render()
    boards, meta = [grab(game)], [(0, 0)]
    done = False
    info = {'perfect': False, 'starved': False, 'score': 0, 'steps': 0}
    while not done:
        action = int(policy_fn(observation.reshape(1, -1))[0])
        observation, _, done, info = env.step(action)
        boards.append(grab(game))
        meta.append((game.current_score, game.current_step))

    # Both endings need the same repair, and for the same reason: `step()` cleared the sprites and
    # `render()` returned before redrawing them, so the frame the game produced is a message on an
    # empty board. Draw the board, then put the message on top of it. The board frame and the
    # message frame are returned as the last two entries so the caller can hold each separately.
    won = game.perfect_game
    if game.finished or won:
        redraw_board(game)
        boards[-1] = grab(game)
        meta[-1] = (game.current_score, game.current_step)
        overlay_win_message(game) if won else overlay_death_message(game)
        boards.append(grab(game))
        meta.append(meta[-1])
    return boards, meta, won, info


def main(argv):
    parser = argparse.ArgumentParser(
        description='Record a policy playing straight to an animated GIF.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('policy', nargs='?', default='hof',
                        help="'hof' for the record, a hallOfFame entry, a savedPolicies name, "
                             'or a path to a checkpoint directory')
    parser.add_argument('step', nargs='?', type=int, default=None,
                        help='checkpoint step (default: the highest in the directory)')
    parser.add_argument('--seconds', type=float, default=60.0,
                        help='minimum playback length; whole games always finish, so the '
                             'recording runs this long or --min-games games, whichever is longer')
    parser.add_argument('--min-games', type=int, default=3,
                        help='complete games to record before the clock is even considered')
    parser.add_argument('--fps', type=float, default=50.0,
                        help='frames per second; a GIF can only do 50, 33.3, 25 or 20')
    parser.add_argument('--tile', type=int, default=30, help='board pixels per tile')
    parser.add_argument('--scale', type=int, default=1, help='integer upscale, nearest-neighbour')
    parser.add_argument('--seed', type=int, default=1, help='first game seed; games run seed, seed+1, ...')
    parser.add_argument('--out', default=None, help='output path (default: gifs/<policy>.<fmt>)')
    parser.add_argument('--format', choices=('gif', 'webp'), default='gif')
    parser.add_argument('--colors', type=int, default=MAX_GIF_COLORS,
                        help='GIF palette size; the board needs six, the rest is text '
                             'antialiasing, and a smaller palette compresses better')
    parser.add_argument('--no-header', action='store_true', help='board only, no header strip')
    parser.add_argument('--allow-losses', action='store_true',
                        help='include games the policy loses (default: keep only perfect games)')
    parser.add_argument('--max-games', type=int, default=40, help='attempts before giving up')
    parser.add_argument('--list', action='store_true', help='list hallOfFame entries and exit')
    args = parser.parse_args(argv[1:])

    if args.min_games > args.max_games:
        raise SystemExit('--min-games %d cannot be met in --max-games %d attempts'
                         % (args.min_games, args.max_games))

    if args.list:
        print('hallOfFame/ (record marked *):')
        for name in hof_entries():
            print('  %s %s' % ('*' if name == HOF_RECORD else ' ', name))
        return 0

    delay_ms = gif_delay_ms(args.fps) if args.format == 'gif' else max(1, int(round(1000.0 / args.fps)))
    if args.format == 'gif' and abs(1000.0 / delay_ms - args.fps) > 0.05:
        print('note: %.4g fps is not representable in a GIF; using %.4g fps (%d ms delay). '
              '--format webp has no such limit.' % (args.fps, 1000.0 / delay_ms, delay_ms))
    target = original_target = target_frame_count(args.seconds, delay_ms)

    ckpt_dir, step, label = resolve_policy(args.policy, args.step)

    # Imported here, not at module scope: torch costs a second to import, and --list and every
    # argument error above should be instant.
    from env.scalar_env import SnakeEnv
    from tools import restore

    disable_display_throttles()
    if not args.no_header:
        suppress_game_hud()

    # display=True with the dummy driver: the drawing happens, the flip does not. policy_name is
    # empty because the header carries the label; the game's own line would be drawn under the
    # sprites. The discount reaches only the shaping potential, which nothing greedy reads.
    env = SnakeEnv(discount=0.9975, display=True, limit_fps=False, policy_name='')
    policy_fn, _, _ = restore.restore(ckpt_dir, step)
    print('%s: floor is %d game(s) or %d frames (%.4g fps, %d ms/frame, %.1f s), whichever is longer'
          % (label, args.min_games, target, 1000.0 / delay_ms, delay_ms,
             target * delay_ms / 1000.0))

    pad_budget = max(1, int(round(PAD_BUDGET_S * 1000.0 / delay_ms)))
    hold_open = max(1, int(round(HOLD_OPEN_S * 1000.0 / delay_ms)))
    hold_board = max(1, int(round(HOLD_BOARD_S * 1000.0 / delay_ms)))
    hold_message = max(1, int(round(HOLD_MESSAGE_S * 1000.0 / delay_ms)))
    hold_death = max(1, int(round(HOLD_DEATH_S * 1000.0 / delay_ms)))

    boards, meta, game_ids, seeds = [], [], [], []
    games = 0
    started = time.time()
    for attempt in range(args.max_games):
        seed = args.seed + attempt
        random.seed(seed)
        episode_boards, episode_meta, won, info = play_episode(env, policy_fn)
        outcome = 'PERFECT' if won else ('starved' if info['starved'] else 'died')
        print('  seed %d: score %d, %d steps, %s'
              % (seed, info['score'], info['steps'], outcome))
        if not won and not args.allow_losses:
            continue

        games += 1
        seeds.append(seed)
        # The opening board is held only on the first game; between games the finished board and
        # its message are the beat, and a second pause would read as a stall.
        opening = hold_open if games == 1 else 0
        hold_end = hold_message if won else hold_death

        def held(sequence):
            """`sequence` with the opening, the final board and the message each held.

            The last two entries are the finished board and that board with the message over it
            (see `play_episode`), so the board is held first and the message second — without
            that, the completed board is only ever seen through the text.
            """
            return ([sequence[0]] * opening + sequence[:-1]
                    + [sequence[-2]] * max(0, hold_board - 1) + [sequence[-1]] * hold_end)

        segment = held(episode_boards)
        boards.extend(segment)
        meta.extend(held(episode_meta))
        game_ids.extend([games] * len(segment))
        # Both floors have to be met: enough games, and enough seconds. Within a pad budget of
        # the clock counts as meeting it (see PAD_BUDGET_S), because a whole extra game to cover a
        # few frames is a poor trade for a hold nobody can see.
        if games >= args.min_games and len(boards) + pad_budget >= target:
            break

    if not boards:
        raise SystemExit('no game to record: %d attempts, none won. --allow-losses records a '
                         'losing game instead.' % args.max_games)
    captured = len(boards)
    target, pad = fit_length(captured, target, pad_budget)
    if target < original_target:
        print('note: %d frames from %d game(s) is short of the %.1f s floor by more than the '
              '%.1f s pad budget — %d attempts ran out. The recording will run %.1f s.'
              % (captured, games, args.seconds, PAD_BUDGET_S, args.max_games,
                 captured * delay_ms / 1000.0))

    # Never a drop: the recording is whole games at one frame per game step, so the only fitting
    # is holding the final frame, which the encoder merges into one frame with a longer delay.
    indices = list(range(captured)) + [captured - 1] * pad
    fitting = ('whole games, nothing dropped' if not pad else
               'whole games, holding the last frame for %d more (%.2f s)'
               % (pad, pad * delay_ms / 1000.0))
    print('  captured %d frames from %d game(s) in %.1f s; %d frames — %s'
          % (captured, games, time.time() - started, target, fitting))
    composer = Composer(boards[0].size, label, enabled=not args.no_header, scale=args.scale)
    frames = [composer.compose(boards[i], meta[i][0], meta[i][1], game_ids[i], games)
              for i in indices]

    out_path = args.out or os.path.join(GIFS_DIR, '%s.%s'
                                        % (label.replace(' @', '-'), args.format))
    if os.path.dirname(out_path):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    write_animation(frames, out_path, delay_ms, args.format, args.colors)
    print('wrote %s — %d frames, %.1f s, %s, %.2f MB (seeds %s)'
          % (out_path, len(frames), len(frames) * delay_ms / 1000.0,
             '%dx%d' % frames[0].size, os.path.getsize(out_path) / 1e6,
             ','.join(str(s) for s in seeds)))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main(sys.argv))
    except KeyboardInterrupt:
        print('\nstopped')
