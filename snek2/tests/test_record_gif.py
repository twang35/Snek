"""The GIF recorder: the frame-rate grid, duration fitting, and the two rendering fixes.

Written alongside `record_gif.py`. Four of these pin things that are invisible in a diff.

`test_sixty_fps_would_play_at_ten_and_is_refused` is the important one. A GIF stores frame delay
in hundredths of a second, so 60 fps is not representable; the natural rounding gives a 10 ms
delay, and browsers clamp any delay at or below 10 ms to **100 ms**. So the obvious
`int(1000 / fps)` produces a file that plays at 10 fps — *slower* than the 20 fps default and in
the opposite direction from what was asked. Nothing about the resulting file looks wrong: the
delay it contains is exactly what was written. Only a viewer shows it.

`test_overlay_win_message_does_not_erase_the_board` and
`test_redraw_board_draws_what_render_skips` pin the pair of fixes that exist because
`Game.render()` returns early on a win: `step()` has already cleared the sprites, so the game's
own final frame is an empty board with a message on it. The recorder draws the filled board and
then puts the message over it. Both fixes are two lines each and read as redundant — the first
looks like it duplicates what `render()` does, the second like a missing `blit(bg)` — so both are
exactly the kind of thing a later reader removes.

`test_suppress_game_hud_writes_to_the_snake_module` pins *where* the patch lands.
`from snake_constants import *` bound copies at import time, so assigning the colours on
`snake_constants` would leave the drawing code reading the originals and change nothing. The
symptom would be cosmetic and easy to misread as a font problem.
"""
import ast
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Before record_gif, which imports pygame. It sets these itself; a test module that imports it
# second-hand should not depend on being first.
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

from PIL import Image, ImageSequence

import record_gif
import snake_constants
import Snake

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE = os.path.join(REPO, 'record_gif.py')


# ----------------------------- the frame-rate grid ----------------------------- #

def test_the_gif_container_truncates_a_sixty_fps_delay_to_ten_milliseconds():
    """Measured against the encoder, not asserted from arithmetic.

    A caller asking for 60 fps passes `int(1000 / 60)` = 16 ms. The GIF container stores
    hundredths of a second and **truncates**, so the file carries 10 ms — which browsers clamp to
    100 ms, i.e. 10 fps. This writes a real animation and reads the delay back out, so it holds
    whatever Pillow does rather than whatever this comment claims.
    """
    def stored_delay(requested_ms):
        frames = [Image.new('RGB', (16, 16), (i * 60 % 255, 30, 200)) for i in range(4)]
        buffer = io.BytesIO()
        frames[0].save(buffer, format='GIF', save_all=True, append_images=frames[1:],
                       duration=requested_ms, loop=0)
        buffer.seek(0)
        read_back = [f.info.get('duration') for f in ImageSequence.Iterator(Image.open(buffer))]
        return read_back[1]

    assert stored_delay(int(1000.0 / 60)) == 10, (
        'the container no longer truncates; re-derive MIN_SAFE_DELAY_MS')
    assert stored_delay(record_gif.gif_delay_ms(60)) == record_gif.gif_delay_ms(60)


def test_sixty_fps_is_floored_rather_than_delivered_as_ten():
    """The floor is the whole reason `gif_delay_ms` exists rather than `int(1000 / fps)`."""
    assert record_gif.gif_delay_ms(60) == 20, 'a 60 fps request must floor at 20 ms, not 10'
    assert record_gif.gif_delay_ms(100) == 20
    assert record_gif.gif_delay_ms(1000) == 20
    assert record_gif.MIN_SAFE_DELAY_MS > record_gif.GIF_DELAY_UNIT_MS


def test_every_delay_lands_on_the_containers_ten_millisecond_grid():
    """A GIF cannot carry a delay off the 10 ms grid, so producing one is a silent rounding."""
    for fps in (12, 15, 18, 20, 24, 25, 29.97, 30, 33.3, 40, 48, 50, 60, 72, 90, 120):
        delay = record_gif.gif_delay_ms(fps)
        assert delay % record_gif.GIF_DELAY_UNIT_MS == 0, (fps, delay)
        assert delay >= record_gif.MIN_SAFE_DELAY_MS, (fps, delay)


def test_the_representable_rates_round_trip_exactly():
    """The rates a GIF can actually play, and the delays they must produce."""
    for fps, delay in ((50, 20), (33.3, 30), (25, 40), (20, 50), (10, 100)):
        assert record_gif.gif_delay_ms(fps) == delay, fps
        assert abs(1000.0 / delay - fps) < 0.35, fps


def test_target_frame_count_is_the_requested_length():
    assert record_gif.target_frame_count(60, 20) == 3000
    assert record_gif.target_frame_count(60, 50) == 1200
    assert record_gif.target_frame_count(8, 20) == 400
    # Never zero, however short the request: a zero-frame animation cannot be written.
    assert record_gif.target_frame_count(0.001, 50) == 1


# ----------------------------- duration fitting ----------------------------- #

def test_resample_hits_the_target_length_exactly_in_both_directions():
    source = list(range(1000))
    for target in (1, 2, 17, 999, 1000, 1001, 3000):
        got = record_gif.resample(source, target)
        assert len(got) == target, (target, len(got))


def test_resample_keeps_the_first_and_last_frame_and_stays_in_order():
    """The last frame is the win message and the first is the opening board; losing either to a
    rounding would silently cut the beat the recording exists for."""
    source = list(range(1234))
    got = record_gif.resample(source, 800)
    assert got[0] == 0 and got[-1] == 1233
    assert all(b >= a for a, b in zip(got, got[1:])), 'resample must not reorder frames'
    assert set(got) <= set(source)


def test_resample_of_a_single_target_is_the_final_frame():
    assert record_gif.resample(list(range(50)), 1) == [49]


def test_the_pad_budget_is_shorter_than_a_game_and_longer_than_a_rounding():
    """PAD_BUDGET_S exists so a small shortfall is held rather than paid for with a whole extra
    game, which forces a large resample. A budget of zero restores that behaviour; one of many
    seconds would pad a recording with a long freeze."""
    assert 0 < record_gif.PAD_BUDGET_S <= 3.0
    assert record_gif.PAD_BUDGET_S >= record_gif.HOLD_BOARD_S


# ----------------------------- the HUD suppression ----------------------------- #

def test_suppress_game_hud_writes_to_the_snake_module():
    """All three readouts, on `Snake` — the module whose globals the drawing code reads."""
    saved = (Snake.SCORE_COLOR, Snake.STEP_COLOR, Snake.POLICY_COLOR)
    try:
        Snake.SCORE_COLOR = Snake.STEP_COLOR = Snake.POLICY_COLOR = (1, 2, 3)
        record_gif.suppress_game_hud()
        for name in ('SCORE_COLOR', 'STEP_COLOR', 'POLICY_COLOR'):
            assert getattr(Snake, name) == Snake.BACKGROUND_COLOR, name
    finally:
        Snake.SCORE_COLOR, Snake.STEP_COLOR, Snake.POLICY_COLOR = saved


def test_the_hud_is_drawn_before_the_sprites_which_is_why_it_has_to_be_hidden():
    """The reason a header is composited outside the board instead of using the game's HUD.

    If `render()` ever drew the readouts *after* `self.all.draw`, they would sit on top of the
    snake, be legible, and the suppression here would be throwing away something useful.
    """
    source = ast.parse(open(os.path.join(REPO, 'Snake.py'), encoding='utf-8').read())
    render = next(n for n in ast.walk(source)
                  if isinstance(n, ast.FunctionDef) and n.name == 'render')
    body = ast.dump(render)
    hud = body.index("'SCORE_PREFIX'") if "'SCORE_PREFIX'" in body else body.index('SCORE_PREFIX')
    sprites = body.index("attr='draw'")
    assert hud < sprites, 'the HUD is no longer drawn under the sprites; revisit suppress_game_hud'


# ----------------------------- the two win-frame fixes ----------------------------- #

def _fresh_game():
    game = Snake.Game(display=True, limit_fps=False, policy_name='')
    game.reset()
    return game


def test_redraw_board_draws_what_render_skips():
    """`render()` returns before drawing the sprites on a winning step, so the recorder draws
    them. Deleting the `all.draw` line leaves the frame blank and this fails."""
    game = _fresh_game()
    game.screen.blit(game.bg, (0, 0))
    blank = record_gif.grab(game)
    record_gif.redraw_board(game)
    drawn = record_gif.grab(game)
    assert drawn.tobytes() != blank.tobytes(), 'redraw_board drew nothing'
    differing = sum(1 for a, b in zip(blank.getdata(), drawn.getdata()) if a != b)
    assert differing > 0


def test_overlay_win_message_does_not_erase_the_board():
    """The message goes *over* the finished board. The game's own version blits onto a cleared
    screen, which is why its win frame is a white card; a `blit(bg)` creeping back in here would
    reproduce that, and this fixture is what catches it."""
    game = _fresh_game()
    board_colour = (255, 0, 0)
    game.screen.fill(board_colour)
    before = record_gif.grab(game)
    record_gif.overlay_win_message(game)
    after = record_gif.grab(game)

    assert after.tobytes() != before.tobytes(), 'no message was drawn'
    width, height = after.size
    for corner in ((0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)):
        assert after.getpixel(corner) == board_colour, (
            'the board was erased at %s: the message must be overlaid, not blitted onto a '
            'cleared screen' % (corner,))
    # And the text really is text: some pixels went dark.
    assert any(sum(p) < 200 for p in after.getdata()), 'the message left no dark pixels'


def test_the_recorded_win_frame_order_shows_the_board_before_the_message():
    """Two held beats, in this order. Reversing them, or dropping the board hold, means the
    finished board is only ever seen under the text."""
    assert record_gif.HOLD_BOARD_S > 0 and record_gif.HOLD_MESSAGE_S > 0
    source = ast.parse(open(SOURCE, encoding='utf-8').read())
    play = next(n for n in ast.walk(source)
                if isinstance(n, ast.FunctionDef) and n.name == 'play_episode')
    calls = [n.func.id for n in ast.walk(play)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    assert calls.index('redraw_board') < calls.index('overlay_win_message')


# ----------------------------- headless-ness and the shortcut ----------------------------- #

def test_the_death_message_is_overlaid_on_the_board_not_on_a_cleared_screen():
    """A death has the same blank-card bug as a win: `render()` returns early there too, so the
    game's own final frame is 'DED' on white and the board showing *where* it died is never
    drawn. The repair, and this fixture, mirror the winning pair exactly."""
    game = _fresh_game()
    board_colour = (255, 0, 0)
    game.screen.fill(board_colour)
    game.starved = False
    before = record_gif.grab(game)
    record_gif.overlay_death_message(game)
    after = record_gif.grab(game)

    assert after.tobytes() != before.tobytes(), 'no death message was drawn'
    width, height = after.size
    for corner in ((0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)):
        assert after.getpixel(corner) == board_colour, (
            'the board was erased at %s; the death message must be overlaid' % (corner,))


def test_the_death_message_matches_the_games_own_choice_of_text_and_size():
    """'NO FUD' when starved, 'DED' otherwise, at the game's two sizes. Inventing a third message
    here would make a recording that does not look like the game."""
    game = _fresh_game()
    seen = []
    real_font = game._message_font

    def recording_font(text, size):
        seen.append((text, size))
        return real_font(text, size)

    game._message_font = recording_font
    game.starved = True
    record_gif.overlay_death_message(game)
    game.starved = False
    record_gif.overlay_death_message(game)

    assert [text for text, _ in seen] == ['NO FUD', 'DED'], seen
    assert seen[0][1] == snake_constants.STARVE_FONT_SIZE
    assert seen[1][1] == snake_constants.DEATH_FONT_SIZE
    assert seen[0][1] != seen[1][1], 'the two messages are sized differently by the game'


def test_both_endings_are_held_long_enough_to_be_seen():
    """One frame is 20 ms at the default rate, which is invisible. Every ending gets a real beat.

    The death hold is deliberately shorter than the win's: with --allow-losses there can be
    several deaths, and a death is punctuation rather than the payoff.
    """
    for hold in (record_gif.HOLD_DEATH_S, record_gif.HOLD_BOARD_S, record_gif.HOLD_MESSAGE_S):
        assert hold >= 0.3, hold
    assert record_gif.HOLD_DEATH_S <= record_gif.HOLD_MESSAGE_S


def test_play_episode_repairs_a_death_as_well_as_a_win():
    """Both branches, behind one `game.finished` guard. Losing the death branch would put a blank
    'DED' card into any recording made with --allow-losses."""
    source = ast.parse(open(SOURCE, encoding='utf-8').read())
    play = next(n for n in ast.walk(source)
                if isinstance(n, ast.FunctionDef) and n.name == 'play_episode')
    called = {n.func.id for n in ast.walk(play)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert {'redraw_board', 'overlay_win_message', 'overlay_death_message'} <= called
    attributes = {n.attr for n in ast.walk(play) if isinstance(n, ast.Attribute)}
    assert 'finished' in attributes, 'the repair must be guarded on the episode having ended'


def test_a_recording_longer_than_the_floor_keeps_every_frame():
    """"3 games or 60 seconds, whichever is longer": three champion games are 80-95 s, so the
    clock is usually not the binding floor and the recording simply runs long. Thinning it back to
    60 s would make it a time-lapse of three games rather than a recording of them."""
    for captured in (3001, 4334, 4633, 9000):
        length, pad = record_gif.fit_length(captured, 3000, 100)
        assert length == captured and pad == 0, (captured, length, pad)


def test_no_recording_is_ever_shortened_to_fit_the_clock():
    """The invariant, over the whole space: fitting may lengthen a recording, never shorten it."""
    for captured in range(1, 400, 7):
        for target in range(1, 400, 11):
            for pad_budget in (0, 1, 25, 100):
                length, pad = record_gif.fit_length(captured, target, pad_budget)
                assert length >= captured, (captured, target, pad_budget, length)
                assert pad == length - captured


def test_a_small_shortfall_is_padded_and_a_large_one_runs_short():
    """Padding is for a rounding, not for a missing game: the pad budget is the line between a
    hold nobody sees and a freeze."""
    length, pad = record_gif.fit_length(2935, 3000, 100)
    assert (length, pad) == (3000, 65), 'a 65-frame shortfall is inside a 100-frame budget'
    length, pad = record_gif.fit_length(900, 3000, 100)
    assert (length, pad) == (900, 0), 'a missing game must not become a 42 s freeze'


def test_three_complete_games_is_the_default_floor():
    """The number the user asked for. A default of 1 would make --seconds the only floor again."""
    source = ast.parse(open(SOURCE, encoding='utf-8').read())
    defaults = {}
    for node in ast.walk(source):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'add_argument' and node.args
                and isinstance(node.args[0], ast.Constant)):
            for keyword in node.keywords:
                if keyword.arg == 'default' and isinstance(keyword.value, ast.Constant):
                    defaults[node.args[0].value] = keyword.value.value
    assert defaults.get('--min-games') == 3, defaults.get('--min-games')
    assert defaults.get('--seconds') == 60.0, defaults.get('--seconds')


def test_both_floors_gate_the_capture_loop():
    """The stop condition has to test the game count *and* the frame count. Dropping either one
    silently restores the previous behaviour, which nothing else here would catch."""
    source = ast.parse(open(SOURCE, encoding='utf-8').read())
    main = next(n for n in ast.walk(source)
                if isinstance(n, ast.FunctionDef) and n.name == 'main')
    breaking = [n for n in ast.walk(main)
                if isinstance(n, ast.If) and any(isinstance(b, ast.Break) for b in n.body)]
    conditions = [ast.dump(n.test) for n in breaking]
    assert any('min_games' in c and 'pad_budget' in c for c in conditions), conditions


def _gradient_frames(count=4, size=48):
    """Frames with far more colours than any palette under test.

    Flat frames would make the assertions below vacuous: a six-colour image satisfies a cap of 8
    and a cap of 255 equally, so the fixture would pass with the knob ignored. It did — this is
    the second version.
    """
    frames = []
    for k in range(count):
        frame = Image.new('RGB', (size, size))
        pixels = frame.load()
        for y in range(size):
            for x in range(size):
                pixels[x, y] = ((x * 5 + k) % 256, (y * 5) % 256, (x + y) % 256)
        frames.append(frame)
    return frames


def test_the_palette_knob_actually_bounds_the_palette():
    """A gallery turns this down because the board is six flat colours and the rest is text
    antialiasing. Frame count dominates the file size, so this is a trim, not the lever."""
    frames = _gradient_frames()
    assert len(set(frames[0].getdata())) > 64, 'the fixture needs frames that must be reduced'
    for colors in (2, 8, 32):
        quantised = record_gif.to_shared_palette(frames, colors)
        assert len(quantised) == len(frames)
        for frame in quantised:
            assert frame.mode == 'P'
            assert len(set(frame.getdata())) <= colors, colors


def test_a_large_palette_request_is_not_silently_reduced_to_a_small_one():
    """The other half: the cap has to be the *requested* number, not a constant. Without this a
    knob hardwired to 8 would pass every bound above."""
    frames = _gradient_frames()
    wide = record_gif.to_shared_palette(frames, record_gif.MAX_GIF_COLORS)
    assert len(set(wide[0].getdata())) > 32, 'the palette request is being ignored downwards'


def test_the_palette_request_cannot_exceed_what_a_gif_can_hold():
    frames = [Image.new('RGB', (8, 8), (10, 20, 30)) for _ in range(3)]
    for frame in record_gif.to_shared_palette(frames, 4096):
        assert len(set(frame.getdata())) <= record_gif.MAX_GIF_COLORS
    assert record_gif.MAX_GIF_COLORS <= 255


def test_the_dummy_driver_is_selected_before_pygame_is_imported():
    """The capture is headless *because* of this ordering, and an import moved above it would open
    a real window on the recording host instead of failing."""
    source = ast.parse(open(SOURCE, encoding='utf-8').read())
    driver_line = pygame_line = None
    for node in ast.walk(source):
        if (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Subscript)
                and isinstance(node.targets[0].slice, ast.Constant)
                and node.targets[0].slice.value == 'SDL_VIDEODRIVER'):
            driver_line = node.lineno
        if isinstance(node, ast.Import) and pygame_line is None:
            if any(a.name in ('pygame', 'snake_constants', 'Snake') for a in node.names):
                pygame_line = node.lineno
    assert driver_line is not None, 'record_gif no longer selects a video driver'
    assert pygame_line is not None
    assert driver_line < pygame_line, 'pygame is imported before the dummy driver is selected'


def test_the_hof_record_shortcut_resolves_to_a_loadable_checkpoint():
    """`record_gif.py hof` takes no arguments, so the shortcut has to name something real. It is a
    convenience over `hallOfFame/HOF.md`, which stays the authority on which entry is the
    record — this only pins that the shortcut has not rotted."""
    directory, step, label = record_gif.resolve_policy('hof', None)
    assert os.path.isdir(directory)
    assert os.path.exists(os.path.join(directory, 'arch.json')), (
        'a checkpoint without arch.json cannot be restored; see CLAUDE.md on the sidecar')
    assert os.path.exists(os.path.join(directory, 'ckpt-%d.index' % step))
    assert str(step // 1000) in label


def test_a_missing_policy_names_the_hall_of_fame_instead_of_raising_a_traceback():
    try:
        record_gif.resolve_policy('no-such-arm-anywhere', None)
    except SystemExit as exit_error:
        assert 'hallOfFame' in str(exit_error)
    else:
        raise AssertionError('resolve_policy accepted a directory that does not exist')
