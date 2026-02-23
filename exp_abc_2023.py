#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2023.2.3),
    on February 23, 2026, at 16:50
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

import psychopy
psychopy.useVersion('2023.2.3')


# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '0'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# Store info about the experiment session
psychopyVersion = '2023.2.3'
expName = '01_resting_state'  # from the Builder filename that created this script
expInfo = {
    'participant': '99',
    'session': '01',
    'date': data.getDateStr(),  # add a simple timestamp
    'expName': expName,
    'psychopyVersion': psychopyVersion,
}


def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # temporarily remove keys which the dialog doesn't need to show
    poppedKeys = {
        'date': expInfo.pop('date', data.getDateStr()),
        'expName': expInfo.pop('expName', expName),
        'psychopyVersion': expInfo.pop('psychopyVersion', psychopyVersion),
    }
    # show participant info dialog
    dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # restore hidden keys
    expInfo.update(poppedKeys)
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Code\\github\\breathwork-cognitive-psychopy\\exp_abc_2023.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # this outputs to the screen, not a file
    logging.console.setLevel(logging.EXP)
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log', level=logging.EXP)
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=[1536, 864], fullscr=True, screen=0,
            winType='pyglet', allowStencil=True,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height'
        )
        if expInfo is not None:
            # store frame rate of monitor if we can measure it
            expInfo['frameRate'] = win.getActualFrameRate()
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    win.mouseVisible = False
    win.hideMessage()
    return win


def setupInputs(expInfo, thisExp, win):
    """
    Setup whatever inputs are available (mouse, keyboard, eyetracker, etc.)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    dict
        Dictionary of input devices by name.
    """
    # --- Setup input devices ---
    inputs = {}
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    ioSession = '1'
    if 'session' in expInfo:
        ioSession = str(expInfo['session'])
    ioServer = io.launchHubServer(window=win, **ioConfig)
    eyetracker = None
    
    # create a default keyboard (e.g. to check for escape)
    defaultKeyboard = keyboard.Keyboard(backend='iohub')
    # return inputs dict
    return {
        'ioServer': ioServer,
        'defaultKeyboard': defaultKeyboard,
        'eyetracker': eyetracker,
    }

def pauseExperiment(thisExp, inputs=None, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # prevent components from auto-drawing
    win.stashAutoDraw()
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # make sure we have a keyboard
        if inputs is None:
            inputs = {
                'defaultKeyboard': keyboard.Keyboard(backend='ioHub')
            }
        # check for quit (typically the Esc key)
        if inputs['defaultKeyboard'].getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win, inputs=inputs)
        # flip the screen
        win.flip()
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, inputs=inputs, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # restore auto-drawn components
    win.retrieveAutoDraw()
    # reset any timers
    for timer in timers:
        timer.reset()


def run(expInfo, thisExp, win, inputs, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    inputs : dict
        Dictionary of input devices by name.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = inputs['ioServer']
    defaultKeyboard = inputs['defaultKeyboard']
    eyetracker = inputs['eyetracker']
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "Trigger_setup" ---
    # Run 'Begin Experiment' code from Trigger_setup_2
    from pylsl import StreamInfo, StreamOutlet # import required classes
    
    info = StreamInfo(name='Trigger',type='Markers', channel_count=1,
    channel_format='int32', source_id='Example') # sets variables for object info
    
    outlet = StreamOutlet(info) # initialise stream.
    
    # --- Initialize components for Routine "Experiment_Setup" ---
    # Run 'Begin Experiment' code from load_data
    """
    Author: Robert Zsolt Kabai
    Email: kabairobert@gmail.com
    Date: May 25, 2025
    Version: 1.0.0
    License: GNU General Public License (GPLv3)
    """
    
    import pandas as pd
    from psychopy import core, visual, event, sound
    
    # Read the Excel file
    schedule_data = pd.read_excel('participant_schedule.xlsx')
    
    # Get current participant and session numbers
    participant_num = int(expInfo['participant'])
    session_num = int(expInfo['session'])
    
    # Find the row for the current participant and session
    current_row = schedule_data[(schedule_data['participant'] == participant_num) & 
                                (schedule_data['session'] == session_num)]
    
    if not current_row.empty:
        # Get video filenames for the current participant and session
        interview_video = current_row['interview_video'].values[0]
        breathing_video = current_row['breathing_protocol'].values[0]
        
        # Store these in expInfo for easier access
        expInfo['interview_video'] = interview_video
        expInfo['breathing_video'] = breathing_video
    else:
        # Handle case where participant/session combination is not in the Excel file
        print(f"Warning: Participant {participant_num}, Session {session_num} not found in participant_schedule.xlsx")
        # Set default video filenames
        expInfo['interview_video'] = 'default_interview.mp4'
        expInfo['breathing_video'] = 'resting_state'
    
    instructed_breathing_instruction = " task: 6 min. \n Just follow the video for inhale and exhale, and relax. \n Press F1 to start."
    resting_state_instruction = " task: 6 min. \n Please keep looking at the cross for 6 minutes. \n Press F1 to start."
    
    breathing_task_text = ''
    if breathing_video == 'paced_breathing_360.mp4':
        breathing_task_text = 'Paced Breathing' + instructed_breathing_instruction
    elif breathing_video == 'fast_with_breath_hold_360.mp4':
        breathing_task_text = 'Fast Paced Breathing with Breath Hold' + instructed_breathing_instruction
    elif breathing_video == 'resting_state_360.mp4':
        breathing_task_text = 'Resting State' + resting_state_instruction
    #breathing_task_text = breathing_task_text + " task: 6 min. \n Just follow the video for inhale and exhale, and relax. \n Press F1 to start."
    
    # Print debug information
    print(f"Participant: {participant_num}, Session: {session_num}")
    print(f"Current row: {current_row}")
    print(f"Breathing video: {expInfo['breathing_video']}")
    print(f"Interview video: {expInfo['interview_video']}")
    
    # Print all data stored in expInfo
    print(f"All data in expInfo: {expInfo}")
    
    # Initialize the wrong answer sound
    #wrong_sound = sound.Sound('sound/wrong.wav')
    # Replace the current wrong_sound initialization with:
    wrong_sound = sound.Sound('sound/wrong.wav')
    # Set experiment start values for variable component breath_duration
    breath_duration = 5
    breath_durationContainer = []
    # Set experiment start values for variable component stress_duration
    stress_duration = 5
    stress_durationContainer = []
    
    # --- Initialize components for Routine "_01_08_rest_state_instr" ---
    text = visual.TextStim(win=win, name='text',
        text='Resting State task\nPlease keep looking at the cross for 6 minutes \nwithout thinking about anything in particular.\n\nPress F1 to start.',
        font='Open Sans',
        pos=(0, 0), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp = keyboard.Keyboard()
    # Run 'Begin Experiment' code from t_begin
    rest_state_instr_counter = -6
    
    # --- Initialize components for Routine "_01_08_rest_state_trial" ---
    # Run 'Begin Experiment' code from t_stim
    rest_state_counter = -60
    cross = visual.TextStim(win=win, name='cross',
        text='+',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "task_end_question" ---
    # Run 'Begin Experiment' code from code_questionaire
    question_counter = 0
    question_text = 'Questionnaire #'
    text_question_number = visual.TextStim(win=win, name='text_question_number',
        text='',
        font='Open Sans',
        pos=[0, 0.4], height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    text_q_stress = visual.TextStim(win=win, name='text_q_stress',
        text='What is your current stress/calm level?\n(-10: as stressed as possible,  0: neutral, +10: as calm as possible)\n\n\n\n\n\nWhat is your current cap comfort level?\n(0: super uncomfortable, 10: super comfortable)\n\n\n\n\n\n\n\nPress F1 to continue.',
        font='Open Sans',
        pos=(0, 0), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    exp_end_bip = sound.Sound('sound/bip.wav', secs=1.0, stereo=False, hamming=True,
        name='exp_end_bip')
    exp_end_bip.setVolume(1.0)
    key_q_stress = keyboard.Keyboard()
    slider = visual.Slider(win=win, name='slider',
        startValue=0, size=(1.5, 0.02), pos=(0, 0.18), units=win.units,
        labels=[-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ticks=(-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), granularity=1.0,
        style='rating', styleTweaks=('labels45', 'triangleMarker'), opacity=None,
        labelColor='LightGray', markerColor=[0.0000, 0.0000, 0.0000], lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.03,
        flip=False, ori=0.0, depth=-5, readOnly=False)
    slider2 = visual.Slider(win=win, name='slider2',
        startValue=5, size=(1.5, 0.02), pos=(0, -0.05), units=win.units,
        labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ticks=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), granularity=1.0,
        style='rating', styleTweaks=('labels45', 'triangleMarker'), opacity=None,
        labelColor='LightGray', markerColor=[0.0000, 0.0000, 0.0000], lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.03,
        flip=False, ori=0.0, depth=-6, readOnly=False)
    
    # --- Initialize components for Routine "_02_stroop_instr" ---
    stroop_instr = visual.TextStim(win=win, name='stroop_instr',
        text='Stroop task\nUse the arrow keys to identify the text color. Ignore what the word says.\n\nPress F1 to start.\n',
        font='Arial',
        pos=(0, 0), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    block_resp = keyboard.Keyboard()
    instrText_4 = visual.TextStim(win=win, name='instrText_4',
        text='red = ←, green = ↓, blue = → \nIdentify the COLOR of the text. Ignore what the word says',
        font='Arial',
        units='height', pos=(0, 0.4), height=0.03, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "_02_stroop_trial" ---
    stim = visual.TextStim(win=win, name='stim',
        text='',
        font='Arial',
        units='height', pos=(0, 0), height=0.15, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-1.0);
    resp = keyboard.Keyboard()
    trial_counter = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Arial',
         pos=(0, -.4),     letterHeight=0.05,
         size=(0.5, 0.1), borderWidth=2.0,
         color='black', colorSpace='rgb',
         opacity=0.8,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor='white', borderColor='black',
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='trial_counter',
         depth=-3, autoLog=True,
    )
    instrText_2 = visual.TextStim(win=win, name='instrText_2',
        text='red = ←, green = ↓, blue = → \nIdentify the COLOR of the text. Ignore what the word says',
        font='Arial',
        units='height', pos=(0, 0.4), height=0.03, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-4.0);
    
    # --- Initialize components for Routine "_02_stroop_feedback" ---
    # Run 'Begin Experiment' code from code_2
    english_accuracy = []
    #maori_accuracy = []
    fbtxt = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Arial',
         pos=(0, 0),     letterHeight=0.05,
         size=(0.5, 0.5), borderWidth=2.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='fbtxt',
         depth=-2, autoLog=True,
    )
    trial_counter_2 = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Arial',
         pos=(0, -.4),     letterHeight=0.05,
         size=(0.5, 0.1), borderWidth=2.0,
         color='black', colorSpace='rgb',
         opacity=0.8,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor='white', borderColor='black',
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='trial_counter_2',
         depth=-3, autoLog=True,
    )
    instrText_3 = visual.TextStim(win=win, name='instrText_3',
        text='red = ←, green = ↓, blue = → \nIdentify the COLOR of the text. Ignore what the word says',
        font='Arial',
        units='height', pos=(0, 0.4), height=0.03, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-4.0);
    
    # --- Initialize components for Routine "_02_stroop_end" ---
    
    # --- Initialize components for Routine "task_end_question" ---
    # Run 'Begin Experiment' code from code_questionaire
    question_counter = 0
    question_text = 'Questionnaire #'
    text_question_number = visual.TextStim(win=win, name='text_question_number',
        text='',
        font='Open Sans',
        pos=[0, 0.4], height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    text_q_stress = visual.TextStim(win=win, name='text_q_stress',
        text='What is your current stress/calm level?\n(-10: as stressed as possible,  0: neutral, +10: as calm as possible)\n\n\n\n\n\nWhat is your current cap comfort level?\n(0: super uncomfortable, 10: super comfortable)\n\n\n\n\n\n\n\nPress F1 to continue.',
        font='Open Sans',
        pos=(0, 0), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    exp_end_bip = sound.Sound('sound/bip.wav', secs=1.0, stereo=False, hamming=True,
        name='exp_end_bip')
    exp_end_bip.setVolume(1.0)
    key_q_stress = keyboard.Keyboard()
    slider = visual.Slider(win=win, name='slider',
        startValue=0, size=(1.5, 0.02), pos=(0, 0.18), units=win.units,
        labels=[-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ticks=(-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), granularity=1.0,
        style='rating', styleTweaks=('labels45', 'triangleMarker'), opacity=None,
        labelColor='LightGray', markerColor=[0.0000, 0.0000, 0.0000], lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.03,
        flip=False, ori=0.0, depth=-5, readOnly=False)
    slider2 = visual.Slider(win=win, name='slider2',
        startValue=5, size=(1.5, 0.02), pos=(0, -0.05), units=win.units,
        labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ticks=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), granularity=1.0,
        style='rating', styleTweaks=('labels45', 'triangleMarker'), opacity=None,
        labelColor='LightGray', markerColor=[0.0000, 0.0000, 0.0000], lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.03,
        flip=False, ori=0.0, depth=-6, readOnly=False)
    
    # --- Initialize components for Routine "_03_05_07_breathing_instr" ---
    text_3 = visual.TextStim(win=win, name='text_3',
        text=breathing_task_text,
        font='Open Sans',
        pos=(0, 0), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_3 = keyboard.Keyboard()
    # Run 'Begin Experiment' code from t_begin_3
    breathing_instr_counter = 1
    
    # --- Initialize components for Routine "_03_05_07_breathing_trial" ---
    # Run 'Begin Experiment' code from t_stim_3
    breathing_counter = 10
    video_breath = visual.MovieStim(
        win, name='video_breath',
        filename=None, movieLib='ffpyplayer',
        loop=False, volume=1.0, noAudio=False,
        pos=(0, 0), size=[1.78, 1.0], units=win.units,
        ori=0.0, anchor='center',opacity=None, contrast=1.0,
        depth=-1
    )
    
    # --- Initialize components for Routine "task_end_question" ---
    # Run 'Begin Experiment' code from code_questionaire
    question_counter = 0
    question_text = 'Questionnaire #'
    text_question_number = visual.TextStim(win=win, name='text_question_number',
        text='',
        font='Open Sans',
        pos=[0, 0.4], height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    text_q_stress = visual.TextStim(win=win, name='text_q_stress',
        text='What is your current stress/calm level?\n(-10: as stressed as possible,  0: neutral, +10: as calm as possible)\n\n\n\n\n\nWhat is your current cap comfort level?\n(0: super uncomfortable, 10: super comfortable)\n\n\n\n\n\n\n\nPress F1 to continue.',
        font='Open Sans',
        pos=(0, 0), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    exp_end_bip = sound.Sound('sound/bip.wav', secs=1.0, stereo=False, hamming=True,
        name='exp_end_bip')
    exp_end_bip.setVolume(1.0)
    key_q_stress = keyboard.Keyboard()
    slider = visual.Slider(win=win, name='slider',
        startValue=0, size=(1.5, 0.02), pos=(0, 0.18), units=win.units,
        labels=[-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ticks=(-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), granularity=1.0,
        style='rating', styleTweaks=('labels45', 'triangleMarker'), opacity=None,
        labelColor='LightGray', markerColor=[0.0000, 0.0000, 0.0000], lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.03,
        flip=False, ori=0.0, depth=-5, readOnly=False)
    slider2 = visual.Slider(win=win, name='slider2',
        startValue=5, size=(1.5, 0.02), pos=(0, -0.05), units=win.units,
        labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ticks=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), granularity=1.0,
        style='rating', styleTweaks=('labels45', 'triangleMarker'), opacity=None,
        labelColor='LightGray', markerColor=[0.0000, 0.0000, 0.0000], lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.03,
        flip=False, ori=0.0, depth=-6, readOnly=False)
    
    # --- Initialize components for Routine "_04_arithmetic_instr" ---
    # Run 'Begin Experiment' code from instr_arithmetic
    from psychopy import visual
    from psychopy import event # For event.waitKeys()
    
    
    # --- Initialize components for Routine "_04_arithmetic_trial" ---
    
    # --- Initialize components for Routine "task_end_question" ---
    # Run 'Begin Experiment' code from code_questionaire
    question_counter = 0
    question_text = 'Questionnaire #'
    text_question_number = visual.TextStim(win=win, name='text_question_number',
        text='',
        font='Open Sans',
        pos=[0, 0.4], height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    text_q_stress = visual.TextStim(win=win, name='text_q_stress',
        text='What is your current stress/calm level?\n(-10: as stressed as possible,  0: neutral, +10: as calm as possible)\n\n\n\n\n\nWhat is your current cap comfort level?\n(0: super uncomfortable, 10: super comfortable)\n\n\n\n\n\n\n\nPress F1 to continue.',
        font='Open Sans',
        pos=(0, 0), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    exp_end_bip = sound.Sound('sound/bip.wav', secs=1.0, stereo=False, hamming=True,
        name='exp_end_bip')
    exp_end_bip.setVolume(1.0)
    key_q_stress = keyboard.Keyboard()
    slider = visual.Slider(win=win, name='slider',
        startValue=0, size=(1.5, 0.02), pos=(0, 0.18), units=win.units,
        labels=[-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ticks=(-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), granularity=1.0,
        style='rating', styleTweaks=('labels45', 'triangleMarker'), opacity=None,
        labelColor='LightGray', markerColor=[0.0000, 0.0000, 0.0000], lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.03,
        flip=False, ori=0.0, depth=-5, readOnly=False)
    slider2 = visual.Slider(win=win, name='slider2',
        startValue=5, size=(1.5, 0.02), pos=(0, -0.05), units=win.units,
        labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ticks=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), granularity=1.0,
        style='rating', styleTweaks=('labels45', 'triangleMarker'), opacity=None,
        labelColor='LightGray', markerColor=[0.0000, 0.0000, 0.0000], lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.03,
        flip=False, ori=0.0, depth=-6, readOnly=False)
    
    # --- Initialize components for Routine "_03_05_07_breathing_instr" ---
    text_3 = visual.TextStim(win=win, name='text_3',
        text=breathing_task_text,
        font='Open Sans',
        pos=(0, 0), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_3 = keyboard.Keyboard()
    # Run 'Begin Experiment' code from t_begin_3
    breathing_instr_counter = 1
    
    # --- Initialize components for Routine "_03_05_07_breathing_trial" ---
    # Run 'Begin Experiment' code from t_stim_3
    breathing_counter = 10
    video_breath = visual.MovieStim(
        win, name='video_breath',
        filename=None, movieLib='ffpyplayer',
        loop=False, volume=1.0, noAudio=False,
        pos=(0, 0), size=[1.78, 1.0], units=win.units,
        ori=0.0, anchor='center',opacity=None, contrast=1.0,
        depth=-1
    )
    
    # --- Initialize components for Routine "task_end_question" ---
    # Run 'Begin Experiment' code from code_questionaire
    question_counter = 0
    question_text = 'Questionnaire #'
    text_question_number = visual.TextStim(win=win, name='text_question_number',
        text='',
        font='Open Sans',
        pos=[0, 0.4], height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    text_q_stress = visual.TextStim(win=win, name='text_q_stress',
        text='What is your current stress/calm level?\n(-10: as stressed as possible,  0: neutral, +10: as calm as possible)\n\n\n\n\n\nWhat is your current cap comfort level?\n(0: super uncomfortable, 10: super comfortable)\n\n\n\n\n\n\n\nPress F1 to continue.',
        font='Open Sans',
        pos=(0, 0), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    exp_end_bip = sound.Sound('sound/bip.wav', secs=1.0, stereo=False, hamming=True,
        name='exp_end_bip')
    exp_end_bip.setVolume(1.0)
    key_q_stress = keyboard.Keyboard()
    slider = visual.Slider(win=win, name='slider',
        startValue=0, size=(1.5, 0.02), pos=(0, 0.18), units=win.units,
        labels=[-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ticks=(-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), granularity=1.0,
        style='rating', styleTweaks=('labels45', 'triangleMarker'), opacity=None,
        labelColor='LightGray', markerColor=[0.0000, 0.0000, 0.0000], lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.03,
        flip=False, ori=0.0, depth=-5, readOnly=False)
    slider2 = visual.Slider(win=win, name='slider2',
        startValue=5, size=(1.5, 0.02), pos=(0, -0.05), units=win.units,
        labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ticks=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), granularity=1.0,
        style='rating', styleTweaks=('labels45', 'triangleMarker'), opacity=None,
        labelColor='LightGray', markerColor=[0.0000, 0.0000, 0.0000], lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.03,
        flip=False, ori=0.0, depth=-6, readOnly=False)
    
    # --- Initialize components for Routine "_06_pubspeak_instr" ---
    text_2 = visual.TextStim(win=win, name='text_2',
        text='Job Interview: 3 min.\nSpeak as in a real interview. \nTry to speak fluently, but no "umh"  filler words! \nEvery "umh" filler word will be notified!\n\nPress F1 to start.',
        font='Open Sans',
        pos=(0, 0), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_2 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "_06_pubspeak_trial" ---
    video = visual.MovieStim(
        win, name='video',
        filename=None, movieLib='ffpyplayer',
        loop=False, volume=1.0, noAudio=False,
        pos=(0, 0), size=[1.78, 1.0], units=win.units,
        ori=0.0, anchor='center',opacity=None, contrast=1.0,
        depth=-1
    )
    
    # --- Initialize components for Routine "task_end_question" ---
    # Run 'Begin Experiment' code from code_questionaire
    question_counter = 0
    question_text = 'Questionnaire #'
    text_question_number = visual.TextStim(win=win, name='text_question_number',
        text='',
        font='Open Sans',
        pos=[0, 0.4], height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    text_q_stress = visual.TextStim(win=win, name='text_q_stress',
        text='What is your current stress/calm level?\n(-10: as stressed as possible,  0: neutral, +10: as calm as possible)\n\n\n\n\n\nWhat is your current cap comfort level?\n(0: super uncomfortable, 10: super comfortable)\n\n\n\n\n\n\n\nPress F1 to continue.',
        font='Open Sans',
        pos=(0, 0), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    exp_end_bip = sound.Sound('sound/bip.wav', secs=1.0, stereo=False, hamming=True,
        name='exp_end_bip')
    exp_end_bip.setVolume(1.0)
    key_q_stress = keyboard.Keyboard()
    slider = visual.Slider(win=win, name='slider',
        startValue=0, size=(1.5, 0.02), pos=(0, 0.18), units=win.units,
        labels=[-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ticks=(-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), granularity=1.0,
        style='rating', styleTweaks=('labels45', 'triangleMarker'), opacity=None,
        labelColor='LightGray', markerColor=[0.0000, 0.0000, 0.0000], lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.03,
        flip=False, ori=0.0, depth=-5, readOnly=False)
    slider2 = visual.Slider(win=win, name='slider2',
        startValue=5, size=(1.5, 0.02), pos=(0, -0.05), units=win.units,
        labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ticks=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), granularity=1.0,
        style='rating', styleTweaks=('labels45', 'triangleMarker'), opacity=None,
        labelColor='LightGray', markerColor=[0.0000, 0.0000, 0.0000], lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.03,
        flip=False, ori=0.0, depth=-6, readOnly=False)
    
    # --- Initialize components for Routine "_03_05_07_breathing_instr" ---
    text_3 = visual.TextStim(win=win, name='text_3',
        text=breathing_task_text,
        font='Open Sans',
        pos=(0, 0), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_3 = keyboard.Keyboard()
    # Run 'Begin Experiment' code from t_begin_3
    breathing_instr_counter = 1
    
    # --- Initialize components for Routine "_03_05_07_breathing_trial" ---
    # Run 'Begin Experiment' code from t_stim_3
    breathing_counter = 10
    video_breath = visual.MovieStim(
        win, name='video_breath',
        filename=None, movieLib='ffpyplayer',
        loop=False, volume=1.0, noAudio=False,
        pos=(0, 0), size=[1.78, 1.0], units=win.units,
        ori=0.0, anchor='center',opacity=None, contrast=1.0,
        depth=-1
    )
    
    # --- Initialize components for Routine "task_end_question" ---
    # Run 'Begin Experiment' code from code_questionaire
    question_counter = 0
    question_text = 'Questionnaire #'
    text_question_number = visual.TextStim(win=win, name='text_question_number',
        text='',
        font='Open Sans',
        pos=[0, 0.4], height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    text_q_stress = visual.TextStim(win=win, name='text_q_stress',
        text='What is your current stress/calm level?\n(-10: as stressed as possible,  0: neutral, +10: as calm as possible)\n\n\n\n\n\nWhat is your current cap comfort level?\n(0: super uncomfortable, 10: super comfortable)\n\n\n\n\n\n\n\nPress F1 to continue.',
        font='Open Sans',
        pos=(0, 0), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    exp_end_bip = sound.Sound('sound/bip.wav', secs=1.0, stereo=False, hamming=True,
        name='exp_end_bip')
    exp_end_bip.setVolume(1.0)
    key_q_stress = keyboard.Keyboard()
    slider = visual.Slider(win=win, name='slider',
        startValue=0, size=(1.5, 0.02), pos=(0, 0.18), units=win.units,
        labels=[-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ticks=(-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), granularity=1.0,
        style='rating', styleTweaks=('labels45', 'triangleMarker'), opacity=None,
        labelColor='LightGray', markerColor=[0.0000, 0.0000, 0.0000], lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.03,
        flip=False, ori=0.0, depth=-5, readOnly=False)
    slider2 = visual.Slider(win=win, name='slider2',
        startValue=5, size=(1.5, 0.02), pos=(0, -0.05), units=win.units,
        labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ticks=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), granularity=1.0,
        style='rating', styleTweaks=('labels45', 'triangleMarker'), opacity=None,
        labelColor='LightGray', markerColor=[0.0000, 0.0000, 0.0000], lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.03,
        flip=False, ori=0.0, depth=-6, readOnly=False)
    
    # --- Initialize components for Routine "_01_08_rest_state_instr" ---
    text = visual.TextStim(win=win, name='text',
        text='Resting State task\nPlease keep looking at the cross for 6 minutes \nwithout thinking about anything in particular.\n\nPress F1 to start.',
        font='Open Sans',
        pos=(0, 0), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp = keyboard.Keyboard()
    # Run 'Begin Experiment' code from t_begin
    rest_state_instr_counter = -6
    
    # --- Initialize components for Routine "_01_08_rest_state_trial" ---
    # Run 'Begin Experiment' code from t_stim
    rest_state_counter = -60
    cross = visual.TextStim(win=win, name='cross',
        text='+',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "task_end_question" ---
    # Run 'Begin Experiment' code from code_questionaire
    question_counter = 0
    question_text = 'Questionnaire #'
    text_question_number = visual.TextStim(win=win, name='text_question_number',
        text='',
        font='Open Sans',
        pos=[0, 0.4], height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    text_q_stress = visual.TextStim(win=win, name='text_q_stress',
        text='What is your current stress/calm level?\n(-10: as stressed as possible,  0: neutral, +10: as calm as possible)\n\n\n\n\n\nWhat is your current cap comfort level?\n(0: super uncomfortable, 10: super comfortable)\n\n\n\n\n\n\n\nPress F1 to continue.',
        font='Open Sans',
        pos=(0, 0), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    exp_end_bip = sound.Sound('sound/bip.wav', secs=1.0, stereo=False, hamming=True,
        name='exp_end_bip')
    exp_end_bip.setVolume(1.0)
    key_q_stress = keyboard.Keyboard()
    slider = visual.Slider(win=win, name='slider',
        startValue=0, size=(1.5, 0.02), pos=(0, 0.18), units=win.units,
        labels=[-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ticks=(-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), granularity=1.0,
        style='rating', styleTweaks=('labels45', 'triangleMarker'), opacity=None,
        labelColor='LightGray', markerColor=[0.0000, 0.0000, 0.0000], lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.03,
        flip=False, ori=0.0, depth=-5, readOnly=False)
    slider2 = visual.Slider(win=win, name='slider2',
        startValue=5, size=(1.5, 0.02), pos=(0, -0.05), units=win.units,
        labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ticks=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), granularity=1.0,
        style='rating', styleTweaks=('labels45', 'triangleMarker'), opacity=None,
        labelColor='LightGray', markerColor=[0.0000, 0.0000, 0.0000], lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.03,
        flip=False, ori=0.0, depth=-6, readOnly=False)
    
    # --- Initialize components for Routine "End" ---
    EndMessage = visual.TextStim(win=win, name='EndMessage',
        text='Thank you!',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    thankyou = sound.Sound('sound/thankyou.wav', secs=1.0, stereo=False, hamming=True,
        name='thankyou')
    thankyou.setVolume(1.0)
    
    # create some handy timers
    if globalClock is None:
        globalClock = core.Clock()  # to track the time since experiment started
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    routineTimer = core.Clock()  # to track time remaining of each (possibly non-slip) routine
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6)
    
    # --- Prepare to start Routine "Trigger_setup" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('Trigger_setup.started', globalClock.getTime())
    # keep track of which components have finished
    Trigger_setupComponents = []
    for thisComponent in Trigger_setupComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Trigger_setup" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Trigger_setupComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Trigger_setup" ---
    for thisComponent in Trigger_setupComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('Trigger_setup.stopped', globalClock.getTime())
    # the Routine "Trigger_setup" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "Experiment_Setup" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('Experiment_Setup.started', globalClock.getTime())
    # keep track of which components have finished
    Experiment_SetupComponents = []
    for thisComponent in Experiment_SetupComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Experiment_Setup" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Experiment_SetupComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Experiment_Setup" ---
    for thisComponent in Experiment_SetupComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('Experiment_Setup.stopped', globalClock.getTime())
    thisExp.addData('breath_duration.expStartVal', 5)  # Save exp start value
    
    # the Routine "Experiment_Setup" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "_01_08_rest_state_instr" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('_01_08_rest_state_instr.started', globalClock.getTime())
    key_resp.keys = []
    key_resp.rt = []
    _key_resp_allKeys = []
    # Run 'Begin Routine' code from t_begin
    rest_state_instr_counter += 7
    outlet.push_sample(x=[rest_state_instr_counter])
    # keep track of which components have finished
    _01_08_rest_state_instrComponents = [text, key_resp]
    for thisComponent in _01_08_rest_state_instrComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "_01_08_rest_state_instr" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text* updates
        
        # if text is starting this frame...
        if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text.frameNStart = frameN  # exact frame index
            text.tStart = t  # local t and not account for scr refresh
            text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text.started')
            # update status
            text.status = STARTED
            text.setAutoDraw(True)
        
        # if text is active this frame...
        if text.status == STARTED:
            # update params
            pass
        
        # *key_resp* updates
        waitOnFlip = False
        
        # if key_resp is starting this frame...
        if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp.frameNStart = frameN  # exact frame index
            key_resp.tStart = t  # local t and not account for scr refresh
            key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp.started')
            # update status
            key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp.status == STARTED and not waitOnFlip:
            theseKeys = key_resp.getKeys(keyList=['f1'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_allKeys.extend(theseKeys)
            if len(_key_resp_allKeys):
                key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                key_resp.rt = _key_resp_allKeys[-1].rt
                key_resp.duration = _key_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in _01_08_rest_state_instrComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "_01_08_rest_state_instr" ---
    for thisComponent in _01_08_rest_state_instrComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('_01_08_rest_state_instr.stopped', globalClock.getTime())
    # check responses
    if key_resp.keys in ['', [], None]:  # No response was made
        key_resp.keys = None
    thisExp.addData('key_resp.keys',key_resp.keys)
    if key_resp.keys != None:  # we had a response
        thisExp.addData('key_resp.rt', key_resp.rt)
        thisExp.addData('key_resp.duration', key_resp.duration)
    thisExp.nextEntry()
    # the Routine "_01_08_rest_state_instr" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "_01_08_rest_state_trial" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('_01_08_rest_state_trial.started', globalClock.getTime())
    # Run 'Begin Routine' code from t_stim
    rest_state_counter += 70
    outlet.push_sample(x=[rest_state_counter+1])
    # keep track of which components have finished
    _01_08_rest_state_trialComponents = [cross]
    for thisComponent in _01_08_rest_state_trialComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "_01_08_rest_state_trial" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *cross* updates
        
        # if cross is starting this frame...
        if cross.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            cross.frameNStart = frameN  # exact frame index
            cross.tStart = t  # local t and not account for scr refresh
            cross.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(cross, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'cross.started')
            # update status
            cross.status = STARTED
            cross.setAutoDraw(True)
        
        # if cross is active this frame...
        if cross.status == STARTED:
            # update params
            pass
        
        # if cross is stopping this frame...
        if cross.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > cross.tStartRefresh + breath_duration-frameTolerance:
                # keep track of stop time/frame for later
                cross.tStop = t  # not accounting for scr refresh
                cross.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cross.stopped')
                # update status
                cross.status = FINISHED
                cross.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in _01_08_rest_state_trialComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "_01_08_rest_state_trial" ---
    for thisComponent in _01_08_rest_state_trialComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('_01_08_rest_state_trial.stopped', globalClock.getTime())
    # Run 'End Routine' code from t_stim
    outlet.push_sample(x=[rest_state_counter+9])
    # the Routine "_01_08_rest_state_trial" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "task_end_question" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('task_end_question.started', globalClock.getTime())
    # Run 'Begin Routine' code from code_questionaire
    question_counter += 1
    current_letter = chr(64 + question_counter)  # 65 is ASCII for 'A'
    question_text = 'Questionnaire: ' + current_letter
    
    outlet.push_sample(x=[100+question_counter])
    text_question_number.setText(question_text)
    exp_end_bip.setSound('sound/bip.wav', secs=1.0, hamming=True)
    exp_end_bip.setVolume(1.0, log=False)
    exp_end_bip.seek(0)
    key_q_stress.keys = []
    key_q_stress.rt = []
    _key_q_stress_allKeys = []
    slider.reset()
    slider2.reset()
    # Run 'Begin Routine' code from code
    slider.marker.size = (0.01, 0.01)
    slider2.marker.size = (0.01, 0.01)
    
    # keep track of which components have finished
    task_end_questionComponents = [text_question_number, text_q_stress, exp_end_bip, key_q_stress, slider, slider2]
    for thisComponent in task_end_questionComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "task_end_question" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_question_number* updates
        
        # if text_question_number is starting this frame...
        if text_question_number.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_question_number.frameNStart = frameN  # exact frame index
            text_question_number.tStart = t  # local t and not account for scr refresh
            text_question_number.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_question_number, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_question_number.started')
            # update status
            text_question_number.status = STARTED
            text_question_number.setAutoDraw(True)
        
        # if text_question_number is active this frame...
        if text_question_number.status == STARTED:
            # update params
            pass
        
        # *text_q_stress* updates
        
        # if text_q_stress is starting this frame...
        if text_q_stress.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_q_stress.frameNStart = frameN  # exact frame index
            text_q_stress.tStart = t  # local t and not account for scr refresh
            text_q_stress.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_q_stress, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_q_stress.started')
            # update status
            text_q_stress.status = STARTED
            text_q_stress.setAutoDraw(True)
        
        # if text_q_stress is active this frame...
        if text_q_stress.status == STARTED:
            # update params
            pass
        
        # if exp_end_bip is starting this frame...
        if exp_end_bip.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            exp_end_bip.frameNStart = frameN  # exact frame index
            exp_end_bip.tStart = t  # local t and not account for scr refresh
            exp_end_bip.tStartRefresh = tThisFlipGlobal  # on global time
            # add timestamp to datafile
            thisExp.addData('exp_end_bip.started', tThisFlipGlobal)
            # update status
            exp_end_bip.status = STARTED
            exp_end_bip.play(when=win)  # sync with win flip
        
        # if exp_end_bip is stopping this frame...
        if exp_end_bip.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > exp_end_bip.tStartRefresh + 1.0-frameTolerance:
                # keep track of stop time/frame for later
                exp_end_bip.tStop = t  # not accounting for scr refresh
                exp_end_bip.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'exp_end_bip.stopped')
                # update status
                exp_end_bip.status = FINISHED
                exp_end_bip.stop()
        # update exp_end_bip status according to whether it's playing
        if exp_end_bip.isPlaying:
            exp_end_bip.status = STARTED
        elif exp_end_bip.isFinished:
            exp_end_bip.status = FINISHED
        
        # *key_q_stress* updates
        waitOnFlip = False
        
        # if key_q_stress is starting this frame...
        if key_q_stress.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_q_stress.frameNStart = frameN  # exact frame index
            key_q_stress.tStart = t  # local t and not account for scr refresh
            key_q_stress.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_q_stress, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_q_stress.started')
            # update status
            key_q_stress.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_q_stress.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_q_stress.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_q_stress.status == STARTED and not waitOnFlip:
            theseKeys = key_q_stress.getKeys(keyList=['f1'], ignoreKeys=["escape"], waitRelease=False)
            _key_q_stress_allKeys.extend(theseKeys)
            if len(_key_q_stress_allKeys):
                key_q_stress.keys = _key_q_stress_allKeys[-1].name  # just the last key pressed
                key_q_stress.rt = _key_q_stress_allKeys[-1].rt
                key_q_stress.duration = _key_q_stress_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *slider* updates
        
        # if slider is starting this frame...
        if slider.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            slider.frameNStart = frameN  # exact frame index
            slider.tStart = t  # local t and not account for scr refresh
            slider.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(slider, 'tStartRefresh')  # time at next scr refresh
            # update status
            slider.status = STARTED
            slider.setAutoDraw(True)
        
        # if slider is active this frame...
        if slider.status == STARTED:
            # update params
            pass
        
        # Check slider for response to end Routine
        if slider.getRating() is not None and slider.status == STARTED:
            continueRoutine = False
        
        # *slider2* updates
        
        # if slider2 is starting this frame...
        if slider2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            slider2.frameNStart = frameN  # exact frame index
            slider2.tStart = t  # local t and not account for scr refresh
            slider2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(slider2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'slider2.started')
            # update status
            slider2.status = STARTED
            slider2.setAutoDraw(True)
        
        # if slider2 is active this frame...
        if slider2.status == STARTED:
            # update params
            pass
        
        # Check slider2 for response to end Routine
        if slider2.getRating() is not None and slider2.status == STARTED:
            continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in task_end_questionComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "task_end_question" ---
    for thisComponent in task_end_questionComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('task_end_question.stopped', globalClock.getTime())
    # Run 'End Routine' code from code_questionaire
    outlet.push_sample(x=[199])
    exp_end_bip.pause()  # ensure sound has stopped at end of Routine
    # check responses
    if key_q_stress.keys in ['', [], None]:  # No response was made
        key_q_stress.keys = None
    thisExp.addData('key_q_stress.keys',key_q_stress.keys)
    if key_q_stress.keys != None:  # we had a response
        thisExp.addData('key_q_stress.rt', key_q_stress.rt)
        thisExp.addData('key_q_stress.duration', key_q_stress.duration)
    thisExp.nextEntry()
    thisExp.addData('slider2.response', slider2.getRating())
    thisExp.addData('slider2.rt', slider2.getRT())
    # the Routine "task_end_question" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "_02_stroop_instr" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('_02_stroop_instr.started', globalClock.getTime())
    block_resp.keys = []
    block_resp.rt = []
    _block_resp_allKeys = []
    # Run 'Begin Routine' code from t_begin_4
    outlet.push_sample(x=[2])
    # keep track of which components have finished
    _02_stroop_instrComponents = [stroop_instr, block_resp, instrText_4]
    for thisComponent in _02_stroop_instrComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "_02_stroop_instr" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *stroop_instr* updates
        
        # if stroop_instr is starting this frame...
        if stroop_instr.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            stroop_instr.frameNStart = frameN  # exact frame index
            stroop_instr.tStart = t  # local t and not account for scr refresh
            stroop_instr.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(stroop_instr, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'stroop_instr.started')
            # update status
            stroop_instr.status = STARTED
            stroop_instr.setAutoDraw(True)
        
        # if stroop_instr is active this frame...
        if stroop_instr.status == STARTED:
            # update params
            pass
        
        # *block_resp* updates
        waitOnFlip = False
        
        # if block_resp is starting this frame...
        if block_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            block_resp.frameNStart = frameN  # exact frame index
            block_resp.tStart = t  # local t and not account for scr refresh
            block_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(block_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'block_resp.started')
            # update status
            block_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(block_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(block_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if block_resp.status == STARTED and not waitOnFlip:
            theseKeys = block_resp.getKeys(keyList=['f1'], ignoreKeys=["escape"], waitRelease=False)
            _block_resp_allKeys.extend(theseKeys)
            if len(_block_resp_allKeys):
                block_resp.keys = _block_resp_allKeys[-1].name  # just the last key pressed
                block_resp.rt = _block_resp_allKeys[-1].rt
                block_resp.duration = _block_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *instrText_4* updates
        
        # if instrText_4 is starting this frame...
        if instrText_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instrText_4.frameNStart = frameN  # exact frame index
            instrText_4.tStart = t  # local t and not account for scr refresh
            instrText_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instrText_4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instrText_4.started')
            # update status
            instrText_4.status = STARTED
            instrText_4.setAutoDraw(True)
        
        # if instrText_4 is active this frame...
        if instrText_4.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in _02_stroop_instrComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "_02_stroop_instr" ---
    for thisComponent in _02_stroop_instrComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('_02_stroop_instr.stopped', globalClock.getTime())
    # check responses
    if block_resp.keys in ['', [], None]:  # No response was made
        block_resp.keys = None
    thisExp.addData('block_resp.keys',block_resp.keys)
    if block_resp.keys != None:  # we had a response
        thisExp.addData('block_resp.rt', block_resp.rt)
        thisExp.addData('block_resp.duration', block_resp.duration)
    thisExp.nextEntry()
    # Run 'End Routine' code from t_begin_4
    outlet.push_sample(x=[21])
    # the Routine "_02_stroop_instr" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler(nReps=1.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('english.xlsx'),
        seed=None, name='trials')
    thisExp.addLoop(trials)  # add the loop to the experiment
    thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            globals()[paramName] = thisTrial[paramName]
    
    for thisTrial in trials:
        currentLoop = trials
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        
        # --- Prepare to start Routine "_02_stroop_trial" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('_02_stroop_trial.started', globalClock.getTime())
        # Run 'Begin Routine' code from t_trial_stroop
        outlet.push_sample(x=[22])
        stim.setColor(wordColor, colorSpace='rgb')
        stim.setText(word)
        resp.keys = []
        resp.rt = []
        _resp_allKeys = []
        trial_counter.reset()
        trial_counter.setText('Trial ' + str(trials.thisN+1) +'/' +str(trials.nTotal))
        # keep track of which components have finished
        _02_stroop_trialComponents = [stim, resp, trial_counter, instrText_2]
        for thisComponent in _02_stroop_trialComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "_02_stroop_trial" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.5:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *stim* updates
            
            # if stim is starting this frame...
            if stim.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                stim.frameNStart = frameN  # exact frame index
                stim.tStart = t  # local t and not account for scr refresh
                stim.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(stim, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'stim.started')
                # update status
                stim.status = STARTED
                stim.setAutoDraw(True)
            
            # if stim is active this frame...
            if stim.status == STARTED:
                # update params
                pass
            
            # if stim is stopping this frame...
            if stim.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > stim.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    stim.tStop = t  # not accounting for scr refresh
                    stim.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'stim.stopped')
                    # update status
                    stim.status = FINISHED
                    stim.setAutoDraw(False)
            
            # *resp* updates
            waitOnFlip = False
            
            # if resp is starting this frame...
            if resp.status == NOT_STARTED and tThisFlip >= .5-frameTolerance:
                # keep track of start time/frame for later
                resp.frameNStart = frameN  # exact frame index
                resp.tStart = t  # local t and not account for scr refresh
                resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'resp.started')
                # update status
                resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if resp is stopping this frame...
            if resp.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > resp.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    resp.tStop = t  # not accounting for scr refresh
                    resp.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'resp.stopped')
                    # update status
                    resp.status = FINISHED
                    resp.status = FINISHED
            if resp.status == STARTED and not waitOnFlip:
                theseKeys = resp.getKeys(keyList=['left','down','right'], ignoreKeys=["escape"], waitRelease=False)
                _resp_allKeys.extend(theseKeys)
                if len(_resp_allKeys):
                    resp.keys = _resp_allKeys[-1].name  # just the last key pressed
                    resp.rt = _resp_allKeys[-1].rt
                    resp.duration = _resp_allKeys[-1].duration
                    # was this correct?
                    if (resp.keys == str(corrAns)) or (resp.keys == corrAns):
                        resp.corr = 1
                    else:
                        resp.corr = 0
                    # a response ends the routine
                    continueRoutine = False
            
            # *trial_counter* updates
            
            # if trial_counter is starting this frame...
            if trial_counter.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                trial_counter.frameNStart = frameN  # exact frame index
                trial_counter.tStart = t  # local t and not account for scr refresh
                trial_counter.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(trial_counter, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'trial_counter.started')
                # update status
                trial_counter.status = STARTED
                trial_counter.setAutoDraw(True)
            
            # if trial_counter is active this frame...
            if trial_counter.status == STARTED:
                # update params
                pass
            
            # if trial_counter is stopping this frame...
            if trial_counter.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > trial_counter.tStartRefresh + 1.5-frameTolerance:
                    # keep track of stop time/frame for later
                    trial_counter.tStop = t  # not accounting for scr refresh
                    trial_counter.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'trial_counter.stopped')
                    # update status
                    trial_counter.status = FINISHED
                    trial_counter.setAutoDraw(False)
            
            # *instrText_2* updates
            
            # if instrText_2 is starting this frame...
            if instrText_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                instrText_2.frameNStart = frameN  # exact frame index
                instrText_2.tStart = t  # local t and not account for scr refresh
                instrText_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(instrText_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'instrText_2.started')
                # update status
                instrText_2.status = STARTED
                instrText_2.setAutoDraw(True)
            
            # if instrText_2 is active this frame...
            if instrText_2.status == STARTED:
                # update params
                pass
            
            # if instrText_2 is stopping this frame...
            if instrText_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > instrText_2.tStartRefresh + 1.5-frameTolerance:
                    # keep track of stop time/frame for later
                    instrText_2.tStop = t  # not accounting for scr refresh
                    instrText_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'instrText_2.stopped')
                    # update status
                    instrText_2.status = FINISHED
                    instrText_2.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in _02_stroop_trialComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "_02_stroop_trial" ---
        for thisComponent in _02_stroop_trialComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('_02_stroop_trial.stopped', globalClock.getTime())
        # check responses
        if resp.keys in ['', [], None]:  # No response was made
            resp.keys = None
            # was no response the correct answer?!
            if str(corrAns).lower() == 'none':
               resp.corr = 1;  # correct non-response
            else:
               resp.corr = 0;  # failed to respond (incorrectly)
        # store data for trials (TrialHandler)
        trials.addData('resp.keys',resp.keys)
        trials.addData('resp.corr', resp.corr)
        if resp.keys != None:  # we had a response
            trials.addData('resp.rt', resp.rt)
            trials.addData('resp.duration', resp.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.500000)
        
        # --- Prepare to start Routine "_02_stroop_feedback" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('_02_stroop_feedback.started', globalClock.getTime())
        # Run 'Begin Routine' code from t_feedback_stroop
        outlet.push_sample(x=[23])
        # Run 'Begin Routine' code from code_2
        if resp.corr:
            fb = 'Correct!'
            fbcol = 'green'
        else:
            fb = 'Incorrect'
            fbcol = 'red'
            #wrong_sound.play()  # Play the wrong answer sound
            try:
                wrong_sound.stop()  # This is critical - must stop before playing again
                core.wait(0.01)     # Brief pause to let the stop take effect
                wrong_sound.play()  # Now play the sound
            except Exception as e:
                print(f"DEBUG: Error with sound: {e}")      
        # track accuracy for each condition by adding a 1 or 0 to a list
        english_accuracy.append(resp.corr)
        
        fbtxt.reset()
        fbtxt.setColor(fbcol, colorSpace='rgb')
        fbtxt.setText(fb)
        trial_counter_2.reset()
        trial_counter_2.setText('Trial ' + str(trials.thisN+1) +'/' +str(trials.nTotal))
        # keep track of which components have finished
        _02_stroop_feedbackComponents = [fbtxt, trial_counter_2, instrText_3]
        for thisComponent in _02_stroop_feedbackComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "_02_stroop_feedback" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.5:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *fbtxt* updates
            
            # if fbtxt is starting this frame...
            if fbtxt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fbtxt.frameNStart = frameN  # exact frame index
                fbtxt.tStart = t  # local t and not account for scr refresh
                fbtxt.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fbtxt, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fbtxt.started')
                # update status
                fbtxt.status = STARTED
                fbtxt.setAutoDraw(True)
            
            # if fbtxt is active this frame...
            if fbtxt.status == STARTED:
                # update params
                pass
            
            # if fbtxt is stopping this frame...
            if fbtxt.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fbtxt.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    fbtxt.tStop = t  # not accounting for scr refresh
                    fbtxt.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fbtxt.stopped')
                    # update status
                    fbtxt.status = FINISHED
                    fbtxt.setAutoDraw(False)
            
            # *trial_counter_2* updates
            
            # if trial_counter_2 is starting this frame...
            if trial_counter_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                trial_counter_2.frameNStart = frameN  # exact frame index
                trial_counter_2.tStart = t  # local t and not account for scr refresh
                trial_counter_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(trial_counter_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'trial_counter_2.started')
                # update status
                trial_counter_2.status = STARTED
                trial_counter_2.setAutoDraw(True)
            
            # if trial_counter_2 is active this frame...
            if trial_counter_2.status == STARTED:
                # update params
                pass
            
            # if trial_counter_2 is stopping this frame...
            if trial_counter_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > trial_counter_2.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    trial_counter_2.tStop = t  # not accounting for scr refresh
                    trial_counter_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'trial_counter_2.stopped')
                    # update status
                    trial_counter_2.status = FINISHED
                    trial_counter_2.setAutoDraw(False)
            
            # *instrText_3* updates
            
            # if instrText_3 is starting this frame...
            if instrText_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                instrText_3.frameNStart = frameN  # exact frame index
                instrText_3.tStart = t  # local t and not account for scr refresh
                instrText_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(instrText_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'instrText_3.started')
                # update status
                instrText_3.status = STARTED
                instrText_3.setAutoDraw(True)
            
            # if instrText_3 is active this frame...
            if instrText_3.status == STARTED:
                # update params
                pass
            
            # if instrText_3 is stopping this frame...
            if instrText_3.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > instrText_3.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    instrText_3.tStop = t  # not accounting for scr refresh
                    instrText_3.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'instrText_3.stopped')
                    # update status
                    instrText_3.status = FINISHED
                    instrText_3.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in _02_stroop_feedbackComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "_02_stroop_feedback" ---
        for thisComponent in _02_stroop_feedbackComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('_02_stroop_feedback.stopped', globalClock.getTime())
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.500000)
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1.0 repeats of 'trials'
    
    
    # --- Prepare to start Routine "_02_stroop_end" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('_02_stroop_end.started', globalClock.getTime())
    # Run 'Begin Routine' code from t_stroop_end
    outlet.push_sample(x=[29])
    # keep track of which components have finished
    _02_stroop_endComponents = []
    for thisComponent in _02_stroop_endComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "_02_stroop_end" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in _02_stroop_endComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "_02_stroop_end" ---
    for thisComponent in _02_stroop_endComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('_02_stroop_end.stopped', globalClock.getTime())
    # the Routine "_02_stroop_end" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "task_end_question" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('task_end_question.started', globalClock.getTime())
    # Run 'Begin Routine' code from code_questionaire
    question_counter += 1
    current_letter = chr(64 + question_counter)  # 65 is ASCII for 'A'
    question_text = 'Questionnaire: ' + current_letter
    
    outlet.push_sample(x=[100+question_counter])
    text_question_number.setText(question_text)
    exp_end_bip.setSound('sound/bip.wav', secs=1.0, hamming=True)
    exp_end_bip.setVolume(1.0, log=False)
    exp_end_bip.seek(0)
    key_q_stress.keys = []
    key_q_stress.rt = []
    _key_q_stress_allKeys = []
    slider.reset()
    slider2.reset()
    # Run 'Begin Routine' code from code
    slider.marker.size = (0.01, 0.01)
    slider2.marker.size = (0.01, 0.01)
    
    # keep track of which components have finished
    task_end_questionComponents = [text_question_number, text_q_stress, exp_end_bip, key_q_stress, slider, slider2]
    for thisComponent in task_end_questionComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "task_end_question" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_question_number* updates
        
        # if text_question_number is starting this frame...
        if text_question_number.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_question_number.frameNStart = frameN  # exact frame index
            text_question_number.tStart = t  # local t and not account for scr refresh
            text_question_number.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_question_number, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_question_number.started')
            # update status
            text_question_number.status = STARTED
            text_question_number.setAutoDraw(True)
        
        # if text_question_number is active this frame...
        if text_question_number.status == STARTED:
            # update params
            pass
        
        # *text_q_stress* updates
        
        # if text_q_stress is starting this frame...
        if text_q_stress.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_q_stress.frameNStart = frameN  # exact frame index
            text_q_stress.tStart = t  # local t and not account for scr refresh
            text_q_stress.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_q_stress, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_q_stress.started')
            # update status
            text_q_stress.status = STARTED
            text_q_stress.setAutoDraw(True)
        
        # if text_q_stress is active this frame...
        if text_q_stress.status == STARTED:
            # update params
            pass
        
        # if exp_end_bip is starting this frame...
        if exp_end_bip.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            exp_end_bip.frameNStart = frameN  # exact frame index
            exp_end_bip.tStart = t  # local t and not account for scr refresh
            exp_end_bip.tStartRefresh = tThisFlipGlobal  # on global time
            # add timestamp to datafile
            thisExp.addData('exp_end_bip.started', tThisFlipGlobal)
            # update status
            exp_end_bip.status = STARTED
            exp_end_bip.play(when=win)  # sync with win flip
        
        # if exp_end_bip is stopping this frame...
        if exp_end_bip.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > exp_end_bip.tStartRefresh + 1.0-frameTolerance:
                # keep track of stop time/frame for later
                exp_end_bip.tStop = t  # not accounting for scr refresh
                exp_end_bip.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'exp_end_bip.stopped')
                # update status
                exp_end_bip.status = FINISHED
                exp_end_bip.stop()
        # update exp_end_bip status according to whether it's playing
        if exp_end_bip.isPlaying:
            exp_end_bip.status = STARTED
        elif exp_end_bip.isFinished:
            exp_end_bip.status = FINISHED
        
        # *key_q_stress* updates
        waitOnFlip = False
        
        # if key_q_stress is starting this frame...
        if key_q_stress.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_q_stress.frameNStart = frameN  # exact frame index
            key_q_stress.tStart = t  # local t and not account for scr refresh
            key_q_stress.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_q_stress, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_q_stress.started')
            # update status
            key_q_stress.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_q_stress.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_q_stress.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_q_stress.status == STARTED and not waitOnFlip:
            theseKeys = key_q_stress.getKeys(keyList=['f1'], ignoreKeys=["escape"], waitRelease=False)
            _key_q_stress_allKeys.extend(theseKeys)
            if len(_key_q_stress_allKeys):
                key_q_stress.keys = _key_q_stress_allKeys[-1].name  # just the last key pressed
                key_q_stress.rt = _key_q_stress_allKeys[-1].rt
                key_q_stress.duration = _key_q_stress_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *slider* updates
        
        # if slider is starting this frame...
        if slider.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            slider.frameNStart = frameN  # exact frame index
            slider.tStart = t  # local t and not account for scr refresh
            slider.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(slider, 'tStartRefresh')  # time at next scr refresh
            # update status
            slider.status = STARTED
            slider.setAutoDraw(True)
        
        # if slider is active this frame...
        if slider.status == STARTED:
            # update params
            pass
        
        # Check slider for response to end Routine
        if slider.getRating() is not None and slider.status == STARTED:
            continueRoutine = False
        
        # *slider2* updates
        
        # if slider2 is starting this frame...
        if slider2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            slider2.frameNStart = frameN  # exact frame index
            slider2.tStart = t  # local t and not account for scr refresh
            slider2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(slider2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'slider2.started')
            # update status
            slider2.status = STARTED
            slider2.setAutoDraw(True)
        
        # if slider2 is active this frame...
        if slider2.status == STARTED:
            # update params
            pass
        
        # Check slider2 for response to end Routine
        if slider2.getRating() is not None and slider2.status == STARTED:
            continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in task_end_questionComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "task_end_question" ---
    for thisComponent in task_end_questionComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('task_end_question.stopped', globalClock.getTime())
    # Run 'End Routine' code from code_questionaire
    outlet.push_sample(x=[199])
    exp_end_bip.pause()  # ensure sound has stopped at end of Routine
    # check responses
    if key_q_stress.keys in ['', [], None]:  # No response was made
        key_q_stress.keys = None
    thisExp.addData('key_q_stress.keys',key_q_stress.keys)
    if key_q_stress.keys != None:  # we had a response
        thisExp.addData('key_q_stress.rt', key_q_stress.rt)
        thisExp.addData('key_q_stress.duration', key_q_stress.duration)
    thisExp.nextEntry()
    thisExp.addData('slider2.response', slider2.getRating())
    thisExp.addData('slider2.rt', slider2.getRT())
    # the Routine "task_end_question" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "_03_05_07_breathing_instr" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('_03_05_07_breathing_instr.started', globalClock.getTime())
    key_resp_3.keys = []
    key_resp_3.rt = []
    _key_resp_3_allKeys = []
    # Run 'Begin Routine' code from t_begin_3
    breathing_instr_counter += 2
    outlet.push_sample(x=[int(breathing_counter)])
    # keep track of which components have finished
    _03_05_07_breathing_instrComponents = [text_3, key_resp_3]
    for thisComponent in _03_05_07_breathing_instrComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "_03_05_07_breathing_instr" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_3* updates
        
        # if text_3 is starting this frame...
        if text_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_3.frameNStart = frameN  # exact frame index
            text_3.tStart = t  # local t and not account for scr refresh
            text_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_3.started')
            # update status
            text_3.status = STARTED
            text_3.setAutoDraw(True)
        
        # if text_3 is active this frame...
        if text_3.status == STARTED:
            # update params
            pass
        
        # *key_resp_3* updates
        waitOnFlip = False
        
        # if key_resp_3 is starting this frame...
        if key_resp_3.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_3.frameNStart = frameN  # exact frame index
            key_resp_3.tStart = t  # local t and not account for scr refresh
            key_resp_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_3.started')
            # update status
            key_resp_3.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_3.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_3.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_3.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_3.getKeys(keyList=['f1'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_3_allKeys.extend(theseKeys)
            if len(_key_resp_3_allKeys):
                key_resp_3.keys = _key_resp_3_allKeys[-1].name  # just the last key pressed
                key_resp_3.rt = _key_resp_3_allKeys[-1].rt
                key_resp_3.duration = _key_resp_3_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in _03_05_07_breathing_instrComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "_03_05_07_breathing_instr" ---
    for thisComponent in _03_05_07_breathing_instrComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('_03_05_07_breathing_instr.stopped', globalClock.getTime())
    # check responses
    if key_resp_3.keys in ['', [], None]:  # No response was made
        key_resp_3.keys = None
    thisExp.addData('key_resp_3.keys',key_resp_3.keys)
    if key_resp_3.keys != None:  # we had a response
        thisExp.addData('key_resp_3.rt', key_resp_3.rt)
        thisExp.addData('key_resp_3.duration', key_resp_3.duration)
    thisExp.nextEntry()
    # the Routine "_03_05_07_breathing_instr" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "_03_05_07_breathing_trial" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('_03_05_07_breathing_trial.started', globalClock.getTime())
    # Run 'Begin Routine' code from t_stim_3
    breathing_counter += 20
    outlet.push_sample(x=[int(breathing_counter)+1])
    video_breath.setMovie(expInfo['breathing_video'])
    # keep track of which components have finished
    _03_05_07_breathing_trialComponents = [video_breath]
    for thisComponent in _03_05_07_breathing_trialComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "_03_05_07_breathing_trial" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *video_breath* updates
        
        # if video_breath is starting this frame...
        if video_breath.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            video_breath.frameNStart = frameN  # exact frame index
            video_breath.tStart = t  # local t and not account for scr refresh
            video_breath.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(video_breath, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'video_breath.started')
            # update status
            video_breath.status = STARTED
            video_breath.setAutoDraw(True)
            video_breath.play()
        
        # if video_breath is stopping this frame...
        if video_breath.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > video_breath.tStartRefresh + breath_duration-frameTolerance:
                # keep track of stop time/frame for later
                video_breath.tStop = t  # not accounting for scr refresh
                video_breath.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'video_breath.stopped')
                # update status
                video_breath.status = FINISHED
                video_breath.setAutoDraw(False)
                video_breath.stop()
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in _03_05_07_breathing_trialComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "_03_05_07_breathing_trial" ---
    for thisComponent in _03_05_07_breathing_trialComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('_03_05_07_breathing_trial.stopped', globalClock.getTime())
    # Run 'End Routine' code from t_stim_3
    outlet.push_sample(x=[int(breathing_counter+9)])
    video_breath.stop()  # ensure movie has stopped at end of Routine
    # the Routine "_03_05_07_breathing_trial" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "task_end_question" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('task_end_question.started', globalClock.getTime())
    # Run 'Begin Routine' code from code_questionaire
    question_counter += 1
    current_letter = chr(64 + question_counter)  # 65 is ASCII for 'A'
    question_text = 'Questionnaire: ' + current_letter
    
    outlet.push_sample(x=[100+question_counter])
    text_question_number.setText(question_text)
    exp_end_bip.setSound('sound/bip.wav', secs=1.0, hamming=True)
    exp_end_bip.setVolume(1.0, log=False)
    exp_end_bip.seek(0)
    key_q_stress.keys = []
    key_q_stress.rt = []
    _key_q_stress_allKeys = []
    slider.reset()
    slider2.reset()
    # Run 'Begin Routine' code from code
    slider.marker.size = (0.01, 0.01)
    slider2.marker.size = (0.01, 0.01)
    
    # keep track of which components have finished
    task_end_questionComponents = [text_question_number, text_q_stress, exp_end_bip, key_q_stress, slider, slider2]
    for thisComponent in task_end_questionComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "task_end_question" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_question_number* updates
        
        # if text_question_number is starting this frame...
        if text_question_number.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_question_number.frameNStart = frameN  # exact frame index
            text_question_number.tStart = t  # local t and not account for scr refresh
            text_question_number.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_question_number, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_question_number.started')
            # update status
            text_question_number.status = STARTED
            text_question_number.setAutoDraw(True)
        
        # if text_question_number is active this frame...
        if text_question_number.status == STARTED:
            # update params
            pass
        
        # *text_q_stress* updates
        
        # if text_q_stress is starting this frame...
        if text_q_stress.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_q_stress.frameNStart = frameN  # exact frame index
            text_q_stress.tStart = t  # local t and not account for scr refresh
            text_q_stress.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_q_stress, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_q_stress.started')
            # update status
            text_q_stress.status = STARTED
            text_q_stress.setAutoDraw(True)
        
        # if text_q_stress is active this frame...
        if text_q_stress.status == STARTED:
            # update params
            pass
        
        # if exp_end_bip is starting this frame...
        if exp_end_bip.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            exp_end_bip.frameNStart = frameN  # exact frame index
            exp_end_bip.tStart = t  # local t and not account for scr refresh
            exp_end_bip.tStartRefresh = tThisFlipGlobal  # on global time
            # add timestamp to datafile
            thisExp.addData('exp_end_bip.started', tThisFlipGlobal)
            # update status
            exp_end_bip.status = STARTED
            exp_end_bip.play(when=win)  # sync with win flip
        
        # if exp_end_bip is stopping this frame...
        if exp_end_bip.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > exp_end_bip.tStartRefresh + 1.0-frameTolerance:
                # keep track of stop time/frame for later
                exp_end_bip.tStop = t  # not accounting for scr refresh
                exp_end_bip.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'exp_end_bip.stopped')
                # update status
                exp_end_bip.status = FINISHED
                exp_end_bip.stop()
        # update exp_end_bip status according to whether it's playing
        if exp_end_bip.isPlaying:
            exp_end_bip.status = STARTED
        elif exp_end_bip.isFinished:
            exp_end_bip.status = FINISHED
        
        # *key_q_stress* updates
        waitOnFlip = False
        
        # if key_q_stress is starting this frame...
        if key_q_stress.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_q_stress.frameNStart = frameN  # exact frame index
            key_q_stress.tStart = t  # local t and not account for scr refresh
            key_q_stress.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_q_stress, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_q_stress.started')
            # update status
            key_q_stress.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_q_stress.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_q_stress.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_q_stress.status == STARTED and not waitOnFlip:
            theseKeys = key_q_stress.getKeys(keyList=['f1'], ignoreKeys=["escape"], waitRelease=False)
            _key_q_stress_allKeys.extend(theseKeys)
            if len(_key_q_stress_allKeys):
                key_q_stress.keys = _key_q_stress_allKeys[-1].name  # just the last key pressed
                key_q_stress.rt = _key_q_stress_allKeys[-1].rt
                key_q_stress.duration = _key_q_stress_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *slider* updates
        
        # if slider is starting this frame...
        if slider.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            slider.frameNStart = frameN  # exact frame index
            slider.tStart = t  # local t and not account for scr refresh
            slider.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(slider, 'tStartRefresh')  # time at next scr refresh
            # update status
            slider.status = STARTED
            slider.setAutoDraw(True)
        
        # if slider is active this frame...
        if slider.status == STARTED:
            # update params
            pass
        
        # Check slider for response to end Routine
        if slider.getRating() is not None and slider.status == STARTED:
            continueRoutine = False
        
        # *slider2* updates
        
        # if slider2 is starting this frame...
        if slider2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            slider2.frameNStart = frameN  # exact frame index
            slider2.tStart = t  # local t and not account for scr refresh
            slider2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(slider2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'slider2.started')
            # update status
            slider2.status = STARTED
            slider2.setAutoDraw(True)
        
        # if slider2 is active this frame...
        if slider2.status == STARTED:
            # update params
            pass
        
        # Check slider2 for response to end Routine
        if slider2.getRating() is not None and slider2.status == STARTED:
            continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in task_end_questionComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "task_end_question" ---
    for thisComponent in task_end_questionComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('task_end_question.stopped', globalClock.getTime())
    # Run 'End Routine' code from code_questionaire
    outlet.push_sample(x=[199])
    exp_end_bip.pause()  # ensure sound has stopped at end of Routine
    # check responses
    if key_q_stress.keys in ['', [], None]:  # No response was made
        key_q_stress.keys = None
    thisExp.addData('key_q_stress.keys',key_q_stress.keys)
    if key_q_stress.keys != None:  # we had a response
        thisExp.addData('key_q_stress.rt', key_q_stress.rt)
        thisExp.addData('key_q_stress.duration', key_q_stress.duration)
    thisExp.nextEntry()
    thisExp.addData('slider2.response', slider2.getRating())
    thisExp.addData('slider2.rt', slider2.getRT())
    # the Routine "task_end_question" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "_04_arithmetic_instr" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('_04_arithmetic_instr.started', globalClock.getTime())
    # Run 'Begin Routine' code from instr_arithmetic
    # Send trigger for initial instructions
    outlet.push_sample(x=[4])
    
    # Initialize components
    instructions = visual.TextStim(win, 
        text="Keep subtracting 13 from the number shown. (count backwards by 13)\nType and press Enter.\nPress F1 to start.",
        height=0.03, color="white", pos=(0, -0.3), wrapWidth=1.5)  # Added wrapWidth
    
    # Show instructions
    instructions.draw()
    win.flip()
    
    # Wait for F1 key press
    event.waitKeys(keyList=['f1'])
    
    # keep track of which components have finished
    _04_arithmetic_instrComponents = []
    for thisComponent in _04_arithmetic_instrComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "_04_arithmetic_instr" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in _04_arithmetic_instrComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "_04_arithmetic_instr" ---
    for thisComponent in _04_arithmetic_instrComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('_04_arithmetic_instr.stopped', globalClock.getTime())
    # the Routine "_04_arithmetic_instr" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "_04_arithmetic_trial" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('_04_arithmetic_trial.started', globalClock.getTime())
    # Run 'Begin Routine' code from code_arithmetic
    """
    Author: Robert Zsolt Kabai
    Email: kabairobert@gmail.com
    Date: May 25, 2025
    Version: 1.0.1
    License: GNU General Public License (GPLv3)
    """
    import random
    
    # Initialize variables
    current_number = random.randint(1000, 1999)
    duration = stress_duration  # Total duration from Builder variable
    last_response_time = 0
    clock = core.Clock()
    start_time = clock.getTime()
    # --- Initialize cumulative stats ---
    total_trials = 0
    total_correct = 0
    total_incorrect = 0
    response_times = []
    
    # --- Define Screen Layout Parameters ---
    screen_width = 2.0  # Width from -1 to 1 in normalized units
    subject_proportion = 2/3  # Subject gets 2/3 of the screen
    experimenter_proportion = 1/3  # Experimenter gets 1/3 of the screen
    
    # Calculate section widths and positions
    subject_width = subject_proportion * screen_width  # 4/3 normalized units
    experimenter_width = experimenter_proportion * screen_width  # 2/3 normalized units
    divider_x_pos = -1 + subject_width  # 1/3 normalized units
    subject_section_center_x = -1 + (subject_width / 2)  # -1/3 normalized units
    experimenter_section_center_x = divider_x_pos + (experimenter_width / 2)  # 2/3 normalized units
    
    # Progress bars should take up 75% of their respective section width
    subject_bar_width = 0.75 * subject_width  # 75% of subject section width
    experimenter_bar_width = 0.75 * experimenter_width  # 75% of experimenter section width
    
    # --- Create Stimuli ---
    divider_line = visual.Line(
        win,
        start=(divider_x_pos, 1),
        end=(divider_x_pos, -1),
        lineColor="white",
        lineWidth=5
    )
    
    text_4digit_subject = visual.TextStim(
        win, 
        text=str(current_number), 
        pos=(subject_section_center_x, 0.2), 
        height=0.1, 
        color="white",
        font='Arial'
    )
    
    text_instruction_subject = visual.TextStim(
        win, 
        text="Subtract 13", 
        pos=(subject_section_center_x, 0.1), 
        height=0.05, 
        color="white",
        font='Arial'
    )
    
    speed_warning_subject = visual.TextStim(
        win, 
        text="FASTER! You're Slow!", 
        pos=(subject_section_center_x, 0.3),
        height=0.08, 
        color="red", 
        bold=True, 
        italic=False, 
        font='Arial'
    )
    
    # New experimenter speed warning text
    speed_warning_experimenter = visual.TextStim(
        win, 
        text="say: FASTER!!", 
        pos=(experimenter_section_center_x, 0.3), 
        height=0.08, 
        color="red", 
        bold=True, 
        italic=False, 
        font='Arial'
    )
    
    text_correct_answer_exp = visual.TextStim(
        win, 
        text="",
        pos=(experimenter_section_center_x, 0.2), 
        height=0.1, 
        color="white",
        font='Arial'
    )
    
    # Create instruction text with arrows
    wrong_instruction = visual.TextStim(
        win, 
        text="← Wrong", 
        pos=(experimenter_section_center_x - 0.08, 0.1),
        height=0.03,
        color="red",
        font='Arial',
        alignText='right',
        wrapWidth=experimenter_bar_width * 0.3
    )
    
    separator_text = visual.TextStim(
        win, 
        text="|", 
        pos=(experimenter_section_center_x, 0.1),
        height=0.03,
        color="white",
        font='Arial',
        alignText='center'
    )
    
    correct_instruction = visual.TextStim(
        win, 
        text="Correct →", 
        pos=(experimenter_section_center_x + 0.08, 0.1),
        height=0.03,
        color="green",
        font='Arial',
        alignText='left',
        wrapWidth=experimenter_bar_width * 0.3
    )
    
    # Feedback text for experimenter and participant sides
    text_exp_feedback = visual.TextStim(
        win, 
        text="", 
        pos=(experimenter_section_center_x, 0.0),
        height=0.07, 
        color="white",
        font='Arial'
    )
    
    text_subject_feedback = visual.TextStim(
        win, 
        text="", 
        pos=(subject_section_center_x, 0.0), 
        height=0.07, 
        color="white",
        font='Arial'
    )
    
    # Subject progress bar (left 2/3 section)
    subject_progress_bar = visual.Rect(
        win, 
        width=subject_bar_width, 
        height=0.05, 
        fillColor="white", 
        lineColor=None, 
        pos=(subject_section_center_x, -0.4)
    )
    
    # Experimenter progress bar (right 1/3 section)
    experimenter_progress_bar = visual.Rect(
        win, 
        width=experimenter_bar_width, 
        height=0.05, 
        fillColor="white", 
        lineColor=None, 
        pos=(experimenter_section_center_x, -0.4)
    )
    
    # Time text for subject section
    time_left_text = visual.TextStim(
        win, 
        text="Time left:", 
        pos=(subject_section_center_x - (subject_bar_width / 2) - 0.05, -0.45), 
        height=0.04, 
        color="white", 
        alignHoriz='left',
        font='Arial'
    )
    
    time_value_text = visual.TextStim(
        win,
        text="",
        pos=(subject_section_center_x - (subject_bar_width / 2) + 0.15, -0.45),
        height=0.04,
        color="white",
        alignHoriz='left',
        font='Arial'
    )
    
    # Time text for experimenter section
    time_left_text_exp = visual.TextStim(
        win, 
        text="Time left:", 
        pos=(experimenter_section_center_x - (experimenter_bar_width / 2) - 0.05, -0.45), 
        height=0.04, 
        color="white", 
        alignHoriz='left',
        font='Arial'
    )
    
    time_value_text_exp = visual.TextStim(
        win,
        text="",
        pos=(experimenter_section_center_x - (experimenter_bar_width / 2) + 0.15, -0.45),
        height=0.04,
        color="white",
        alignHoriz='left',
        font='Arial'
    )
    
    # Trigger for start of task
    if 'outlet' in globals() and outlet is not None:
        outlet.push_sample(x=[41])
    
    # Main loop for task
    while True:
        current_loop_time = clock.getTime()
        time_elapsed = current_loop_time - start_time
        time_left = duration - time_elapsed
        
        if time_left <= 0:
            time_left = 0
            break
        
        trial_start_time = current_loop_time
        
        # if 'outlet' in globals() and outlet is not None:
        #     win.callOnFlip(outlet.push_sample, x=[41])
        
        correct_answer_for_this_trial = current_number - 13
        
        text_4digit_subject.text = str(current_number)
        text_correct_answer_exp.text = f"Ans: {correct_answer_for_this_trial}"
        text_exp_feedback.text = ""
        text_subject_feedback.text = ""  # Clear feedback text
        
        # Calculate progress ratio and update progress bars
        progress_ratio = time_left / duration
        
        # Update subject progress bar
        subject_current_bar_width = progress_ratio * subject_bar_width
        subject_progress_bar.width = subject_current_bar_width
        subject_progress_bar.pos = (
            subject_section_center_x - (subject_bar_width / 2) + (subject_current_bar_width / 2),
            -0.4
        )
        
        # Update experimenter progress bar
        experimenter_current_bar_width = progress_ratio * experimenter_bar_width
        experimenter_progress_bar.width = experimenter_current_bar_width
        experimenter_progress_bar.pos = (
            experimenter_section_center_x - (experimenter_bar_width / 2) + (experimenter_current_bar_width / 2),
            -0.4
        )
        
        # Update time text
        time_value_text.text = f"{max(0, int(time_left))}s"
        time_value_text_exp.text = f"{max(0, int(time_left))}s"
        
        responded_this_trial = False
        experimenter_response_key = None
        event.clearEvents()
        
        while not responded_this_trial:
            current_input_phase_time = clock.getTime()
            time_elapsed_input = current_input_phase_time - start_time
            time_left_input = duration - time_elapsed_input
            
            if time_left_input <= 0:
                time_left = 0
                break
            
            # Calculate progress ratio for this frame
            progress_ratio_input = time_left_input / duration
            
            # Update subject progress bar
            subject_current_bar_width_input = progress_ratio_input * subject_bar_width
            subject_progress_bar.width = subject_current_bar_width_input
            subject_progress_bar.pos = (
                subject_section_center_x - (subject_bar_width / 2) + (subject_current_bar_width_input / 2),
                -0.4
            )
            
            # Update experimenter progress bar
            experimenter_current_bar_width_input = progress_ratio_input * experimenter_bar_width
            experimenter_progress_bar.width = experimenter_current_bar_width_input
            experimenter_progress_bar.pos = (
                experimenter_section_center_x - (experimenter_bar_width / 2) + (experimenter_current_bar_width_input / 2),
                -0.4
            )
            
            # Update time text for both sections
            time_value_text.text = f"{max(0, int(time_left_input))}s"
            time_value_text_exp.text = f"{max(0, int(time_left_input))}s"
            
            # Draw all components
            text_4digit_subject.draw()
            text_instruction_subject.draw()
            
            # Show speed warnings if needed
            if last_response_time >= 3:
                speed_warning_subject.draw()
                speed_warning_experimenter.draw()  # Draw speed warning for experimenter
            
            text_correct_answer_exp.draw()
            
            # Draw the instruction text with arrows
            wrong_instruction.draw()
            separator_text.draw()
            correct_instruction.draw()
            
            divider_line.draw()
            subject_progress_bar.draw()
            experimenter_progress_bar.draw()
            time_left_text.draw()
            time_value_text.draw()
            time_left_text_exp.draw()
            time_value_text_exp.draw()
            
            win.flip()
            
            keys = event.getKeys(keyList=['left', 'right', 'escape'])
            if 'escape' in keys:
                if 'outlet' in globals() and outlet is not None:
                    outlet.push_sample(x=[2]) # Trigger for early exit
                core.quit()
            
            if 'left' in keys:
                experimenter_response_key = 'left'
                responded_this_trial = True
            elif 'right' in keys:
                experimenter_response_key = 'right'
                responded_this_trial = True
            
            if time_left <= 0:
                break
        
        if time_left <= 0:
            break
        
        response_time_for_this_trial = clock.getTime() - trial_start_time
        
        if experimenter_response_key == 'left':
            # Update feedback for both sections
            text_exp_feedback.text = "WRONG"
            text_exp_feedback.color = "red"
            text_subject_feedback.text = "WRONG"
            text_subject_feedback.color = "red"
            
            # Stop sound before playing it again
            try:
                wrong_sound.stop()  # This is critical - must stop before playing again
                core.wait(0.01)     # Brief pause to let the stop take effect
                wrong_sound.play()  # Now play the sound
            except Exception as e:
                print(f"DEBUG: Error with sound: {e}")       
        elif experimenter_response_key == 'right':
            # Update feedback for both sections - CORRECT
            text_exp_feedback.text = "CORRECT"
            text_exp_feedback.color = "green"
            text_subject_feedback.text = "CORRECT"
            text_subject_feedback.color = "green"
        
        # --- Log trial data ---
        response_correct = experimenter_response_key == 'right'
        thisExp.addData('presented_number', current_number)
        thisExp.addData('correct_answer', correct_answer_for_this_trial)
        thisExp.addData('experimenter_response', experimenter_response_key)
        thisExp.addData('response_time', response_time_for_this_trial)
        thisExp.addData('response_correct', response_correct)
        thisExp.nextEntry()
        # --- Update cumulative stats ---
        total_trials += 1
        if response_correct:
            total_correct += 1
        else:
            total_incorrect += 1
        response_times.append(response_time_for_this_trial)
        
        # Draw final state of this trial
        text_4digit_subject.draw()
        text_instruction_subject.draw()
        if last_response_time >= 3:
            speed_warning_subject.draw()
            speed_warning_experimenter.draw()
            
        text_correct_answer_exp.draw()
        
        # Draw the instruction text with arrows
        wrong_instruction.draw()
        separator_text.draw()
        correct_instruction.draw()
        
        text_exp_feedback.draw()
        text_subject_feedback.draw()
        
        divider_line.draw()
        subject_progress_bar.draw()
        experimenter_progress_bar.draw()
        time_left_text.draw()
        time_value_text.draw()
        time_left_text_exp.draw()
        time_value_text_exp.draw()
        
        win.flip()
        core.wait(0.75)
        
        current_number = correct_answer_for_this_trial
        last_response_time = response_time_for_this_trial
    
    
    # Trigger for end of task
    if 'outlet' in globals() and outlet is not None:
        outlet.push_sample(x=[49])
    
    # At the end of your arithmetic task
    if 'wrong_sound' in globals():
        try:
            # Stop the sound but keep the reference for later routines
            wrong_sound.stop()
            # Ensure it's fully stopped
            core.wait(0.01)
        except Exception as e:
            print(f"Error stopping wrong_sound: {e}")
            pass
    
    # At the end of your arithmetic task
    import gc
    gc.collect()  # Force garbage collection
    
    # --- Log cumulative summary stats at end ---
    if total_trials > 0:
        mean_response_time = sum(response_times) / total_trials
    else:
        mean_response_time = 0
    thisExp.addData('summary_total_trials', total_trials)
    thisExp.addData('summary_total_correct', total_correct)
    thisExp.addData('summary_total_incorrect', total_incorrect)
    thisExp.addData('summary_mean_response_time', mean_response_time)
    thisExp.nextEntry()
    
    # Just clear the screen and wait
    win.flip()
    core.wait(2.0)
    
    # keep track of which components have finished
    _04_arithmetic_trialComponents = []
    for thisComponent in _04_arithmetic_trialComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "_04_arithmetic_trial" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in _04_arithmetic_trialComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "_04_arithmetic_trial" ---
    for thisComponent in _04_arithmetic_trialComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('_04_arithmetic_trial.stopped', globalClock.getTime())
    # Run 'End Routine' code from code_arithmetic
    outlet.push_sample(x=[49])
    
    # the Routine "_04_arithmetic_trial" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "task_end_question" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('task_end_question.started', globalClock.getTime())
    # Run 'Begin Routine' code from code_questionaire
    question_counter += 1
    current_letter = chr(64 + question_counter)  # 65 is ASCII for 'A'
    question_text = 'Questionnaire: ' + current_letter
    
    outlet.push_sample(x=[100+question_counter])
    text_question_number.setText(question_text)
    exp_end_bip.setSound('sound/bip.wav', secs=1.0, hamming=True)
    exp_end_bip.setVolume(1.0, log=False)
    exp_end_bip.seek(0)
    key_q_stress.keys = []
    key_q_stress.rt = []
    _key_q_stress_allKeys = []
    slider.reset()
    slider2.reset()
    # Run 'Begin Routine' code from code
    slider.marker.size = (0.01, 0.01)
    slider2.marker.size = (0.01, 0.01)
    
    # keep track of which components have finished
    task_end_questionComponents = [text_question_number, text_q_stress, exp_end_bip, key_q_stress, slider, slider2]
    for thisComponent in task_end_questionComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "task_end_question" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_question_number* updates
        
        # if text_question_number is starting this frame...
        if text_question_number.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_question_number.frameNStart = frameN  # exact frame index
            text_question_number.tStart = t  # local t and not account for scr refresh
            text_question_number.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_question_number, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_question_number.started')
            # update status
            text_question_number.status = STARTED
            text_question_number.setAutoDraw(True)
        
        # if text_question_number is active this frame...
        if text_question_number.status == STARTED:
            # update params
            pass
        
        # *text_q_stress* updates
        
        # if text_q_stress is starting this frame...
        if text_q_stress.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_q_stress.frameNStart = frameN  # exact frame index
            text_q_stress.tStart = t  # local t and not account for scr refresh
            text_q_stress.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_q_stress, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_q_stress.started')
            # update status
            text_q_stress.status = STARTED
            text_q_stress.setAutoDraw(True)
        
        # if text_q_stress is active this frame...
        if text_q_stress.status == STARTED:
            # update params
            pass
        
        # if exp_end_bip is starting this frame...
        if exp_end_bip.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            exp_end_bip.frameNStart = frameN  # exact frame index
            exp_end_bip.tStart = t  # local t and not account for scr refresh
            exp_end_bip.tStartRefresh = tThisFlipGlobal  # on global time
            # add timestamp to datafile
            thisExp.addData('exp_end_bip.started', tThisFlipGlobal)
            # update status
            exp_end_bip.status = STARTED
            exp_end_bip.play(when=win)  # sync with win flip
        
        # if exp_end_bip is stopping this frame...
        if exp_end_bip.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > exp_end_bip.tStartRefresh + 1.0-frameTolerance:
                # keep track of stop time/frame for later
                exp_end_bip.tStop = t  # not accounting for scr refresh
                exp_end_bip.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'exp_end_bip.stopped')
                # update status
                exp_end_bip.status = FINISHED
                exp_end_bip.stop()
        # update exp_end_bip status according to whether it's playing
        if exp_end_bip.isPlaying:
            exp_end_bip.status = STARTED
        elif exp_end_bip.isFinished:
            exp_end_bip.status = FINISHED
        
        # *key_q_stress* updates
        waitOnFlip = False
        
        # if key_q_stress is starting this frame...
        if key_q_stress.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_q_stress.frameNStart = frameN  # exact frame index
            key_q_stress.tStart = t  # local t and not account for scr refresh
            key_q_stress.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_q_stress, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_q_stress.started')
            # update status
            key_q_stress.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_q_stress.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_q_stress.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_q_stress.status == STARTED and not waitOnFlip:
            theseKeys = key_q_stress.getKeys(keyList=['f1'], ignoreKeys=["escape"], waitRelease=False)
            _key_q_stress_allKeys.extend(theseKeys)
            if len(_key_q_stress_allKeys):
                key_q_stress.keys = _key_q_stress_allKeys[-1].name  # just the last key pressed
                key_q_stress.rt = _key_q_stress_allKeys[-1].rt
                key_q_stress.duration = _key_q_stress_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *slider* updates
        
        # if slider is starting this frame...
        if slider.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            slider.frameNStart = frameN  # exact frame index
            slider.tStart = t  # local t and not account for scr refresh
            slider.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(slider, 'tStartRefresh')  # time at next scr refresh
            # update status
            slider.status = STARTED
            slider.setAutoDraw(True)
        
        # if slider is active this frame...
        if slider.status == STARTED:
            # update params
            pass
        
        # Check slider for response to end Routine
        if slider.getRating() is not None and slider.status == STARTED:
            continueRoutine = False
        
        # *slider2* updates
        
        # if slider2 is starting this frame...
        if slider2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            slider2.frameNStart = frameN  # exact frame index
            slider2.tStart = t  # local t and not account for scr refresh
            slider2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(slider2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'slider2.started')
            # update status
            slider2.status = STARTED
            slider2.setAutoDraw(True)
        
        # if slider2 is active this frame...
        if slider2.status == STARTED:
            # update params
            pass
        
        # Check slider2 for response to end Routine
        if slider2.getRating() is not None and slider2.status == STARTED:
            continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in task_end_questionComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "task_end_question" ---
    for thisComponent in task_end_questionComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('task_end_question.stopped', globalClock.getTime())
    # Run 'End Routine' code from code_questionaire
    outlet.push_sample(x=[199])
    exp_end_bip.pause()  # ensure sound has stopped at end of Routine
    # check responses
    if key_q_stress.keys in ['', [], None]:  # No response was made
        key_q_stress.keys = None
    thisExp.addData('key_q_stress.keys',key_q_stress.keys)
    if key_q_stress.keys != None:  # we had a response
        thisExp.addData('key_q_stress.rt', key_q_stress.rt)
        thisExp.addData('key_q_stress.duration', key_q_stress.duration)
    thisExp.nextEntry()
    thisExp.addData('slider2.response', slider2.getRating())
    thisExp.addData('slider2.rt', slider2.getRT())
    # the Routine "task_end_question" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "_03_05_07_breathing_instr" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('_03_05_07_breathing_instr.started', globalClock.getTime())
    key_resp_3.keys = []
    key_resp_3.rt = []
    _key_resp_3_allKeys = []
    # Run 'Begin Routine' code from t_begin_3
    breathing_instr_counter += 2
    outlet.push_sample(x=[int(breathing_counter)])
    # keep track of which components have finished
    _03_05_07_breathing_instrComponents = [text_3, key_resp_3]
    for thisComponent in _03_05_07_breathing_instrComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "_03_05_07_breathing_instr" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_3* updates
        
        # if text_3 is starting this frame...
        if text_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_3.frameNStart = frameN  # exact frame index
            text_3.tStart = t  # local t and not account for scr refresh
            text_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_3.started')
            # update status
            text_3.status = STARTED
            text_3.setAutoDraw(True)
        
        # if text_3 is active this frame...
        if text_3.status == STARTED:
            # update params
            pass
        
        # *key_resp_3* updates
        waitOnFlip = False
        
        # if key_resp_3 is starting this frame...
        if key_resp_3.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_3.frameNStart = frameN  # exact frame index
            key_resp_3.tStart = t  # local t and not account for scr refresh
            key_resp_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_3.started')
            # update status
            key_resp_3.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_3.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_3.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_3.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_3.getKeys(keyList=['f1'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_3_allKeys.extend(theseKeys)
            if len(_key_resp_3_allKeys):
                key_resp_3.keys = _key_resp_3_allKeys[-1].name  # just the last key pressed
                key_resp_3.rt = _key_resp_3_allKeys[-1].rt
                key_resp_3.duration = _key_resp_3_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in _03_05_07_breathing_instrComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "_03_05_07_breathing_instr" ---
    for thisComponent in _03_05_07_breathing_instrComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('_03_05_07_breathing_instr.stopped', globalClock.getTime())
    # check responses
    if key_resp_3.keys in ['', [], None]:  # No response was made
        key_resp_3.keys = None
    thisExp.addData('key_resp_3.keys',key_resp_3.keys)
    if key_resp_3.keys != None:  # we had a response
        thisExp.addData('key_resp_3.rt', key_resp_3.rt)
        thisExp.addData('key_resp_3.duration', key_resp_3.duration)
    thisExp.nextEntry()
    # the Routine "_03_05_07_breathing_instr" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "_03_05_07_breathing_trial" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('_03_05_07_breathing_trial.started', globalClock.getTime())
    # Run 'Begin Routine' code from t_stim_3
    breathing_counter += 20
    outlet.push_sample(x=[int(breathing_counter)+1])
    video_breath.setMovie(expInfo['breathing_video'])
    # keep track of which components have finished
    _03_05_07_breathing_trialComponents = [video_breath]
    for thisComponent in _03_05_07_breathing_trialComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "_03_05_07_breathing_trial" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *video_breath* updates
        
        # if video_breath is starting this frame...
        if video_breath.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            video_breath.frameNStart = frameN  # exact frame index
            video_breath.tStart = t  # local t and not account for scr refresh
            video_breath.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(video_breath, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'video_breath.started')
            # update status
            video_breath.status = STARTED
            video_breath.setAutoDraw(True)
            video_breath.play()
        
        # if video_breath is stopping this frame...
        if video_breath.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > video_breath.tStartRefresh + breath_duration-frameTolerance:
                # keep track of stop time/frame for later
                video_breath.tStop = t  # not accounting for scr refresh
                video_breath.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'video_breath.stopped')
                # update status
                video_breath.status = FINISHED
                video_breath.setAutoDraw(False)
                video_breath.stop()
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in _03_05_07_breathing_trialComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "_03_05_07_breathing_trial" ---
    for thisComponent in _03_05_07_breathing_trialComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('_03_05_07_breathing_trial.stopped', globalClock.getTime())
    # Run 'End Routine' code from t_stim_3
    outlet.push_sample(x=[int(breathing_counter+9)])
    video_breath.stop()  # ensure movie has stopped at end of Routine
    # the Routine "_03_05_07_breathing_trial" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "task_end_question" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('task_end_question.started', globalClock.getTime())
    # Run 'Begin Routine' code from code_questionaire
    question_counter += 1
    current_letter = chr(64 + question_counter)  # 65 is ASCII for 'A'
    question_text = 'Questionnaire: ' + current_letter
    
    outlet.push_sample(x=[100+question_counter])
    text_question_number.setText(question_text)
    exp_end_bip.setSound('sound/bip.wav', secs=1.0, hamming=True)
    exp_end_bip.setVolume(1.0, log=False)
    exp_end_bip.seek(0)
    key_q_stress.keys = []
    key_q_stress.rt = []
    _key_q_stress_allKeys = []
    slider.reset()
    slider2.reset()
    # Run 'Begin Routine' code from code
    slider.marker.size = (0.01, 0.01)
    slider2.marker.size = (0.01, 0.01)
    
    # keep track of which components have finished
    task_end_questionComponents = [text_question_number, text_q_stress, exp_end_bip, key_q_stress, slider, slider2]
    for thisComponent in task_end_questionComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "task_end_question" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_question_number* updates
        
        # if text_question_number is starting this frame...
        if text_question_number.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_question_number.frameNStart = frameN  # exact frame index
            text_question_number.tStart = t  # local t and not account for scr refresh
            text_question_number.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_question_number, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_question_number.started')
            # update status
            text_question_number.status = STARTED
            text_question_number.setAutoDraw(True)
        
        # if text_question_number is active this frame...
        if text_question_number.status == STARTED:
            # update params
            pass
        
        # *text_q_stress* updates
        
        # if text_q_stress is starting this frame...
        if text_q_stress.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_q_stress.frameNStart = frameN  # exact frame index
            text_q_stress.tStart = t  # local t and not account for scr refresh
            text_q_stress.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_q_stress, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_q_stress.started')
            # update status
            text_q_stress.status = STARTED
            text_q_stress.setAutoDraw(True)
        
        # if text_q_stress is active this frame...
        if text_q_stress.status == STARTED:
            # update params
            pass
        
        # if exp_end_bip is starting this frame...
        if exp_end_bip.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            exp_end_bip.frameNStart = frameN  # exact frame index
            exp_end_bip.tStart = t  # local t and not account for scr refresh
            exp_end_bip.tStartRefresh = tThisFlipGlobal  # on global time
            # add timestamp to datafile
            thisExp.addData('exp_end_bip.started', tThisFlipGlobal)
            # update status
            exp_end_bip.status = STARTED
            exp_end_bip.play(when=win)  # sync with win flip
        
        # if exp_end_bip is stopping this frame...
        if exp_end_bip.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > exp_end_bip.tStartRefresh + 1.0-frameTolerance:
                # keep track of stop time/frame for later
                exp_end_bip.tStop = t  # not accounting for scr refresh
                exp_end_bip.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'exp_end_bip.stopped')
                # update status
                exp_end_bip.status = FINISHED
                exp_end_bip.stop()
        # update exp_end_bip status according to whether it's playing
        if exp_end_bip.isPlaying:
            exp_end_bip.status = STARTED
        elif exp_end_bip.isFinished:
            exp_end_bip.status = FINISHED
        
        # *key_q_stress* updates
        waitOnFlip = False
        
        # if key_q_stress is starting this frame...
        if key_q_stress.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_q_stress.frameNStart = frameN  # exact frame index
            key_q_stress.tStart = t  # local t and not account for scr refresh
            key_q_stress.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_q_stress, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_q_stress.started')
            # update status
            key_q_stress.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_q_stress.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_q_stress.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_q_stress.status == STARTED and not waitOnFlip:
            theseKeys = key_q_stress.getKeys(keyList=['f1'], ignoreKeys=["escape"], waitRelease=False)
            _key_q_stress_allKeys.extend(theseKeys)
            if len(_key_q_stress_allKeys):
                key_q_stress.keys = _key_q_stress_allKeys[-1].name  # just the last key pressed
                key_q_stress.rt = _key_q_stress_allKeys[-1].rt
                key_q_stress.duration = _key_q_stress_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *slider* updates
        
        # if slider is starting this frame...
        if slider.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            slider.frameNStart = frameN  # exact frame index
            slider.tStart = t  # local t and not account for scr refresh
            slider.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(slider, 'tStartRefresh')  # time at next scr refresh
            # update status
            slider.status = STARTED
            slider.setAutoDraw(True)
        
        # if slider is active this frame...
        if slider.status == STARTED:
            # update params
            pass
        
        # Check slider for response to end Routine
        if slider.getRating() is not None and slider.status == STARTED:
            continueRoutine = False
        
        # *slider2* updates
        
        # if slider2 is starting this frame...
        if slider2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            slider2.frameNStart = frameN  # exact frame index
            slider2.tStart = t  # local t and not account for scr refresh
            slider2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(slider2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'slider2.started')
            # update status
            slider2.status = STARTED
            slider2.setAutoDraw(True)
        
        # if slider2 is active this frame...
        if slider2.status == STARTED:
            # update params
            pass
        
        # Check slider2 for response to end Routine
        if slider2.getRating() is not None and slider2.status == STARTED:
            continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in task_end_questionComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "task_end_question" ---
    for thisComponent in task_end_questionComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('task_end_question.stopped', globalClock.getTime())
    # Run 'End Routine' code from code_questionaire
    outlet.push_sample(x=[199])
    exp_end_bip.pause()  # ensure sound has stopped at end of Routine
    # check responses
    if key_q_stress.keys in ['', [], None]:  # No response was made
        key_q_stress.keys = None
    thisExp.addData('key_q_stress.keys',key_q_stress.keys)
    if key_q_stress.keys != None:  # we had a response
        thisExp.addData('key_q_stress.rt', key_q_stress.rt)
        thisExp.addData('key_q_stress.duration', key_q_stress.duration)
    thisExp.nextEntry()
    thisExp.addData('slider2.response', slider2.getRating())
    thisExp.addData('slider2.rt', slider2.getRT())
    # the Routine "task_end_question" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "_06_pubspeak_instr" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('_06_pubspeak_instr.started', globalClock.getTime())
    key_resp_2.keys = []
    key_resp_2.rt = []
    _key_resp_2_allKeys = []
    # Run 'Begin Routine' code from t_begin_2
    outlet.push_sample(x=[6])
    # keep track of which components have finished
    _06_pubspeak_instrComponents = [text_2, key_resp_2]
    for thisComponent in _06_pubspeak_instrComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "_06_pubspeak_instr" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_2* updates
        
        # if text_2 is starting this frame...
        if text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_2.frameNStart = frameN  # exact frame index
            text_2.tStart = t  # local t and not account for scr refresh
            text_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_2.started')
            # update status
            text_2.status = STARTED
            text_2.setAutoDraw(True)
        
        # if text_2 is active this frame...
        if text_2.status == STARTED:
            # update params
            pass
        
        # *key_resp_2* updates
        waitOnFlip = False
        
        # if key_resp_2 is starting this frame...
        if key_resp_2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_2.frameNStart = frameN  # exact frame index
            key_resp_2.tStart = t  # local t and not account for scr refresh
            key_resp_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_2.started')
            # update status
            key_resp_2.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_2.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_2.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_2.getKeys(keyList=['f1'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_2_allKeys.extend(theseKeys)
            if len(_key_resp_2_allKeys):
                key_resp_2.keys = _key_resp_2_allKeys[-1].name  # just the last key pressed
                key_resp_2.rt = _key_resp_2_allKeys[-1].rt
                key_resp_2.duration = _key_resp_2_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in _06_pubspeak_instrComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "_06_pubspeak_instr" ---
    for thisComponent in _06_pubspeak_instrComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('_06_pubspeak_instr.stopped', globalClock.getTime())
    # check responses
    if key_resp_2.keys in ['', [], None]:  # No response was made
        key_resp_2.keys = None
    thisExp.addData('key_resp_2.keys',key_resp_2.keys)
    if key_resp_2.keys != None:  # we had a response
        thisExp.addData('key_resp_2.rt', key_resp_2.rt)
        thisExp.addData('key_resp_2.duration', key_resp_2.duration)
    thisExp.nextEntry()
    # the Routine "_06_pubspeak_instr" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "_06_pubspeak_trial" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('_06_pubspeak_trial.started', globalClock.getTime())
    # Run 'Begin Routine' code from t_stim_2
    outlet.push_sample(x=[int(61)])
    video.setMovie(expInfo['interview_video'])
    # keep track of which components have finished
    _06_pubspeak_trialComponents = [video]
    for thisComponent in _06_pubspeak_trialComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "_06_pubspeak_trial" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *video* updates
        
        # if video is starting this frame...
        if video.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            video.frameNStart = frameN  # exact frame index
            video.tStart = t  # local t and not account for scr refresh
            video.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(video, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'video.started')
            # update status
            video.status = STARTED
            video.setAutoDraw(True)
            video.play()
        
        # if video is stopping this frame...
        if video.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > video.tStartRefresh + stress_duration-frameTolerance:
                # keep track of stop time/frame for later
                video.tStop = t  # not accounting for scr refresh
                video.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'video.stopped')
                # update status
                video.status = FINISHED
                video.setAutoDraw(False)
                video.stop()
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in _06_pubspeak_trialComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "_06_pubspeak_trial" ---
    for thisComponent in _06_pubspeak_trialComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('_06_pubspeak_trial.stopped', globalClock.getTime())
    # Run 'End Routine' code from t_stim_2
    outlet.push_sample(x=[int(69)])
    video.stop()  # ensure movie has stopped at end of Routine
    # the Routine "_06_pubspeak_trial" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "task_end_question" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('task_end_question.started', globalClock.getTime())
    # Run 'Begin Routine' code from code_questionaire
    question_counter += 1
    current_letter = chr(64 + question_counter)  # 65 is ASCII for 'A'
    question_text = 'Questionnaire: ' + current_letter
    
    outlet.push_sample(x=[100+question_counter])
    text_question_number.setText(question_text)
    exp_end_bip.setSound('sound/bip.wav', secs=1.0, hamming=True)
    exp_end_bip.setVolume(1.0, log=False)
    exp_end_bip.seek(0)
    key_q_stress.keys = []
    key_q_stress.rt = []
    _key_q_stress_allKeys = []
    slider.reset()
    slider2.reset()
    # Run 'Begin Routine' code from code
    slider.marker.size = (0.01, 0.01)
    slider2.marker.size = (0.01, 0.01)
    
    # keep track of which components have finished
    task_end_questionComponents = [text_question_number, text_q_stress, exp_end_bip, key_q_stress, slider, slider2]
    for thisComponent in task_end_questionComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "task_end_question" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_question_number* updates
        
        # if text_question_number is starting this frame...
        if text_question_number.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_question_number.frameNStart = frameN  # exact frame index
            text_question_number.tStart = t  # local t and not account for scr refresh
            text_question_number.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_question_number, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_question_number.started')
            # update status
            text_question_number.status = STARTED
            text_question_number.setAutoDraw(True)
        
        # if text_question_number is active this frame...
        if text_question_number.status == STARTED:
            # update params
            pass
        
        # *text_q_stress* updates
        
        # if text_q_stress is starting this frame...
        if text_q_stress.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_q_stress.frameNStart = frameN  # exact frame index
            text_q_stress.tStart = t  # local t and not account for scr refresh
            text_q_stress.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_q_stress, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_q_stress.started')
            # update status
            text_q_stress.status = STARTED
            text_q_stress.setAutoDraw(True)
        
        # if text_q_stress is active this frame...
        if text_q_stress.status == STARTED:
            # update params
            pass
        
        # if exp_end_bip is starting this frame...
        if exp_end_bip.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            exp_end_bip.frameNStart = frameN  # exact frame index
            exp_end_bip.tStart = t  # local t and not account for scr refresh
            exp_end_bip.tStartRefresh = tThisFlipGlobal  # on global time
            # add timestamp to datafile
            thisExp.addData('exp_end_bip.started', tThisFlipGlobal)
            # update status
            exp_end_bip.status = STARTED
            exp_end_bip.play(when=win)  # sync with win flip
        
        # if exp_end_bip is stopping this frame...
        if exp_end_bip.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > exp_end_bip.tStartRefresh + 1.0-frameTolerance:
                # keep track of stop time/frame for later
                exp_end_bip.tStop = t  # not accounting for scr refresh
                exp_end_bip.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'exp_end_bip.stopped')
                # update status
                exp_end_bip.status = FINISHED
                exp_end_bip.stop()
        # update exp_end_bip status according to whether it's playing
        if exp_end_bip.isPlaying:
            exp_end_bip.status = STARTED
        elif exp_end_bip.isFinished:
            exp_end_bip.status = FINISHED
        
        # *key_q_stress* updates
        waitOnFlip = False
        
        # if key_q_stress is starting this frame...
        if key_q_stress.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_q_stress.frameNStart = frameN  # exact frame index
            key_q_stress.tStart = t  # local t and not account for scr refresh
            key_q_stress.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_q_stress, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_q_stress.started')
            # update status
            key_q_stress.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_q_stress.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_q_stress.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_q_stress.status == STARTED and not waitOnFlip:
            theseKeys = key_q_stress.getKeys(keyList=['f1'], ignoreKeys=["escape"], waitRelease=False)
            _key_q_stress_allKeys.extend(theseKeys)
            if len(_key_q_stress_allKeys):
                key_q_stress.keys = _key_q_stress_allKeys[-1].name  # just the last key pressed
                key_q_stress.rt = _key_q_stress_allKeys[-1].rt
                key_q_stress.duration = _key_q_stress_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *slider* updates
        
        # if slider is starting this frame...
        if slider.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            slider.frameNStart = frameN  # exact frame index
            slider.tStart = t  # local t and not account for scr refresh
            slider.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(slider, 'tStartRefresh')  # time at next scr refresh
            # update status
            slider.status = STARTED
            slider.setAutoDraw(True)
        
        # if slider is active this frame...
        if slider.status == STARTED:
            # update params
            pass
        
        # Check slider for response to end Routine
        if slider.getRating() is not None and slider.status == STARTED:
            continueRoutine = False
        
        # *slider2* updates
        
        # if slider2 is starting this frame...
        if slider2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            slider2.frameNStart = frameN  # exact frame index
            slider2.tStart = t  # local t and not account for scr refresh
            slider2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(slider2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'slider2.started')
            # update status
            slider2.status = STARTED
            slider2.setAutoDraw(True)
        
        # if slider2 is active this frame...
        if slider2.status == STARTED:
            # update params
            pass
        
        # Check slider2 for response to end Routine
        if slider2.getRating() is not None and slider2.status == STARTED:
            continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in task_end_questionComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "task_end_question" ---
    for thisComponent in task_end_questionComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('task_end_question.stopped', globalClock.getTime())
    # Run 'End Routine' code from code_questionaire
    outlet.push_sample(x=[199])
    exp_end_bip.pause()  # ensure sound has stopped at end of Routine
    # check responses
    if key_q_stress.keys in ['', [], None]:  # No response was made
        key_q_stress.keys = None
    thisExp.addData('key_q_stress.keys',key_q_stress.keys)
    if key_q_stress.keys != None:  # we had a response
        thisExp.addData('key_q_stress.rt', key_q_stress.rt)
        thisExp.addData('key_q_stress.duration', key_q_stress.duration)
    thisExp.nextEntry()
    thisExp.addData('slider2.response', slider2.getRating())
    thisExp.addData('slider2.rt', slider2.getRT())
    # the Routine "task_end_question" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "_03_05_07_breathing_instr" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('_03_05_07_breathing_instr.started', globalClock.getTime())
    key_resp_3.keys = []
    key_resp_3.rt = []
    _key_resp_3_allKeys = []
    # Run 'Begin Routine' code from t_begin_3
    breathing_instr_counter += 2
    outlet.push_sample(x=[int(breathing_counter)])
    # keep track of which components have finished
    _03_05_07_breathing_instrComponents = [text_3, key_resp_3]
    for thisComponent in _03_05_07_breathing_instrComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "_03_05_07_breathing_instr" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_3* updates
        
        # if text_3 is starting this frame...
        if text_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_3.frameNStart = frameN  # exact frame index
            text_3.tStart = t  # local t and not account for scr refresh
            text_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_3.started')
            # update status
            text_3.status = STARTED
            text_3.setAutoDraw(True)
        
        # if text_3 is active this frame...
        if text_3.status == STARTED:
            # update params
            pass
        
        # *key_resp_3* updates
        waitOnFlip = False
        
        # if key_resp_3 is starting this frame...
        if key_resp_3.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_3.frameNStart = frameN  # exact frame index
            key_resp_3.tStart = t  # local t and not account for scr refresh
            key_resp_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_3.started')
            # update status
            key_resp_3.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_3.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_3.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_3.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_3.getKeys(keyList=['f1'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_3_allKeys.extend(theseKeys)
            if len(_key_resp_3_allKeys):
                key_resp_3.keys = _key_resp_3_allKeys[-1].name  # just the last key pressed
                key_resp_3.rt = _key_resp_3_allKeys[-1].rt
                key_resp_3.duration = _key_resp_3_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in _03_05_07_breathing_instrComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "_03_05_07_breathing_instr" ---
    for thisComponent in _03_05_07_breathing_instrComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('_03_05_07_breathing_instr.stopped', globalClock.getTime())
    # check responses
    if key_resp_3.keys in ['', [], None]:  # No response was made
        key_resp_3.keys = None
    thisExp.addData('key_resp_3.keys',key_resp_3.keys)
    if key_resp_3.keys != None:  # we had a response
        thisExp.addData('key_resp_3.rt', key_resp_3.rt)
        thisExp.addData('key_resp_3.duration', key_resp_3.duration)
    thisExp.nextEntry()
    # the Routine "_03_05_07_breathing_instr" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "_03_05_07_breathing_trial" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('_03_05_07_breathing_trial.started', globalClock.getTime())
    # Run 'Begin Routine' code from t_stim_3
    breathing_counter += 20
    outlet.push_sample(x=[int(breathing_counter)+1])
    video_breath.setMovie(expInfo['breathing_video'])
    # keep track of which components have finished
    _03_05_07_breathing_trialComponents = [video_breath]
    for thisComponent in _03_05_07_breathing_trialComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "_03_05_07_breathing_trial" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *video_breath* updates
        
        # if video_breath is starting this frame...
        if video_breath.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            video_breath.frameNStart = frameN  # exact frame index
            video_breath.tStart = t  # local t and not account for scr refresh
            video_breath.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(video_breath, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'video_breath.started')
            # update status
            video_breath.status = STARTED
            video_breath.setAutoDraw(True)
            video_breath.play()
        
        # if video_breath is stopping this frame...
        if video_breath.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > video_breath.tStartRefresh + breath_duration-frameTolerance:
                # keep track of stop time/frame for later
                video_breath.tStop = t  # not accounting for scr refresh
                video_breath.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'video_breath.stopped')
                # update status
                video_breath.status = FINISHED
                video_breath.setAutoDraw(False)
                video_breath.stop()
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in _03_05_07_breathing_trialComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "_03_05_07_breathing_trial" ---
    for thisComponent in _03_05_07_breathing_trialComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('_03_05_07_breathing_trial.stopped', globalClock.getTime())
    # Run 'End Routine' code from t_stim_3
    outlet.push_sample(x=[int(breathing_counter+9)])
    video_breath.stop()  # ensure movie has stopped at end of Routine
    # the Routine "_03_05_07_breathing_trial" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "task_end_question" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('task_end_question.started', globalClock.getTime())
    # Run 'Begin Routine' code from code_questionaire
    question_counter += 1
    current_letter = chr(64 + question_counter)  # 65 is ASCII for 'A'
    question_text = 'Questionnaire: ' + current_letter
    
    outlet.push_sample(x=[100+question_counter])
    text_question_number.setText(question_text)
    exp_end_bip.setSound('sound/bip.wav', secs=1.0, hamming=True)
    exp_end_bip.setVolume(1.0, log=False)
    exp_end_bip.seek(0)
    key_q_stress.keys = []
    key_q_stress.rt = []
    _key_q_stress_allKeys = []
    slider.reset()
    slider2.reset()
    # Run 'Begin Routine' code from code
    slider.marker.size = (0.01, 0.01)
    slider2.marker.size = (0.01, 0.01)
    
    # keep track of which components have finished
    task_end_questionComponents = [text_question_number, text_q_stress, exp_end_bip, key_q_stress, slider, slider2]
    for thisComponent in task_end_questionComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "task_end_question" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_question_number* updates
        
        # if text_question_number is starting this frame...
        if text_question_number.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_question_number.frameNStart = frameN  # exact frame index
            text_question_number.tStart = t  # local t and not account for scr refresh
            text_question_number.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_question_number, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_question_number.started')
            # update status
            text_question_number.status = STARTED
            text_question_number.setAutoDraw(True)
        
        # if text_question_number is active this frame...
        if text_question_number.status == STARTED:
            # update params
            pass
        
        # *text_q_stress* updates
        
        # if text_q_stress is starting this frame...
        if text_q_stress.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_q_stress.frameNStart = frameN  # exact frame index
            text_q_stress.tStart = t  # local t and not account for scr refresh
            text_q_stress.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_q_stress, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_q_stress.started')
            # update status
            text_q_stress.status = STARTED
            text_q_stress.setAutoDraw(True)
        
        # if text_q_stress is active this frame...
        if text_q_stress.status == STARTED:
            # update params
            pass
        
        # if exp_end_bip is starting this frame...
        if exp_end_bip.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            exp_end_bip.frameNStart = frameN  # exact frame index
            exp_end_bip.tStart = t  # local t and not account for scr refresh
            exp_end_bip.tStartRefresh = tThisFlipGlobal  # on global time
            # add timestamp to datafile
            thisExp.addData('exp_end_bip.started', tThisFlipGlobal)
            # update status
            exp_end_bip.status = STARTED
            exp_end_bip.play(when=win)  # sync with win flip
        
        # if exp_end_bip is stopping this frame...
        if exp_end_bip.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > exp_end_bip.tStartRefresh + 1.0-frameTolerance:
                # keep track of stop time/frame for later
                exp_end_bip.tStop = t  # not accounting for scr refresh
                exp_end_bip.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'exp_end_bip.stopped')
                # update status
                exp_end_bip.status = FINISHED
                exp_end_bip.stop()
        # update exp_end_bip status according to whether it's playing
        if exp_end_bip.isPlaying:
            exp_end_bip.status = STARTED
        elif exp_end_bip.isFinished:
            exp_end_bip.status = FINISHED
        
        # *key_q_stress* updates
        waitOnFlip = False
        
        # if key_q_stress is starting this frame...
        if key_q_stress.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_q_stress.frameNStart = frameN  # exact frame index
            key_q_stress.tStart = t  # local t and not account for scr refresh
            key_q_stress.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_q_stress, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_q_stress.started')
            # update status
            key_q_stress.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_q_stress.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_q_stress.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_q_stress.status == STARTED and not waitOnFlip:
            theseKeys = key_q_stress.getKeys(keyList=['f1'], ignoreKeys=["escape"], waitRelease=False)
            _key_q_stress_allKeys.extend(theseKeys)
            if len(_key_q_stress_allKeys):
                key_q_stress.keys = _key_q_stress_allKeys[-1].name  # just the last key pressed
                key_q_stress.rt = _key_q_stress_allKeys[-1].rt
                key_q_stress.duration = _key_q_stress_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *slider* updates
        
        # if slider is starting this frame...
        if slider.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            slider.frameNStart = frameN  # exact frame index
            slider.tStart = t  # local t and not account for scr refresh
            slider.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(slider, 'tStartRefresh')  # time at next scr refresh
            # update status
            slider.status = STARTED
            slider.setAutoDraw(True)
        
        # if slider is active this frame...
        if slider.status == STARTED:
            # update params
            pass
        
        # Check slider for response to end Routine
        if slider.getRating() is not None and slider.status == STARTED:
            continueRoutine = False
        
        # *slider2* updates
        
        # if slider2 is starting this frame...
        if slider2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            slider2.frameNStart = frameN  # exact frame index
            slider2.tStart = t  # local t and not account for scr refresh
            slider2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(slider2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'slider2.started')
            # update status
            slider2.status = STARTED
            slider2.setAutoDraw(True)
        
        # if slider2 is active this frame...
        if slider2.status == STARTED:
            # update params
            pass
        
        # Check slider2 for response to end Routine
        if slider2.getRating() is not None and slider2.status == STARTED:
            continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in task_end_questionComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "task_end_question" ---
    for thisComponent in task_end_questionComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('task_end_question.stopped', globalClock.getTime())
    # Run 'End Routine' code from code_questionaire
    outlet.push_sample(x=[199])
    exp_end_bip.pause()  # ensure sound has stopped at end of Routine
    # check responses
    if key_q_stress.keys in ['', [], None]:  # No response was made
        key_q_stress.keys = None
    thisExp.addData('key_q_stress.keys',key_q_stress.keys)
    if key_q_stress.keys != None:  # we had a response
        thisExp.addData('key_q_stress.rt', key_q_stress.rt)
        thisExp.addData('key_q_stress.duration', key_q_stress.duration)
    thisExp.nextEntry()
    thisExp.addData('slider2.response', slider2.getRating())
    thisExp.addData('slider2.rt', slider2.getRT())
    # the Routine "task_end_question" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "_01_08_rest_state_instr" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('_01_08_rest_state_instr.started', globalClock.getTime())
    key_resp.keys = []
    key_resp.rt = []
    _key_resp_allKeys = []
    # Run 'Begin Routine' code from t_begin
    rest_state_instr_counter += 7
    outlet.push_sample(x=[rest_state_instr_counter])
    # keep track of which components have finished
    _01_08_rest_state_instrComponents = [text, key_resp]
    for thisComponent in _01_08_rest_state_instrComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "_01_08_rest_state_instr" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text* updates
        
        # if text is starting this frame...
        if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text.frameNStart = frameN  # exact frame index
            text.tStart = t  # local t and not account for scr refresh
            text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text.started')
            # update status
            text.status = STARTED
            text.setAutoDraw(True)
        
        # if text is active this frame...
        if text.status == STARTED:
            # update params
            pass
        
        # *key_resp* updates
        waitOnFlip = False
        
        # if key_resp is starting this frame...
        if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp.frameNStart = frameN  # exact frame index
            key_resp.tStart = t  # local t and not account for scr refresh
            key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp.started')
            # update status
            key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp.status == STARTED and not waitOnFlip:
            theseKeys = key_resp.getKeys(keyList=['f1'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_allKeys.extend(theseKeys)
            if len(_key_resp_allKeys):
                key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                key_resp.rt = _key_resp_allKeys[-1].rt
                key_resp.duration = _key_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in _01_08_rest_state_instrComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "_01_08_rest_state_instr" ---
    for thisComponent in _01_08_rest_state_instrComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('_01_08_rest_state_instr.stopped', globalClock.getTime())
    # check responses
    if key_resp.keys in ['', [], None]:  # No response was made
        key_resp.keys = None
    thisExp.addData('key_resp.keys',key_resp.keys)
    if key_resp.keys != None:  # we had a response
        thisExp.addData('key_resp.rt', key_resp.rt)
        thisExp.addData('key_resp.duration', key_resp.duration)
    thisExp.nextEntry()
    # the Routine "_01_08_rest_state_instr" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "_01_08_rest_state_trial" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('_01_08_rest_state_trial.started', globalClock.getTime())
    # Run 'Begin Routine' code from t_stim
    rest_state_counter += 70
    outlet.push_sample(x=[rest_state_counter+1])
    # keep track of which components have finished
    _01_08_rest_state_trialComponents = [cross]
    for thisComponent in _01_08_rest_state_trialComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "_01_08_rest_state_trial" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *cross* updates
        
        # if cross is starting this frame...
        if cross.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            cross.frameNStart = frameN  # exact frame index
            cross.tStart = t  # local t and not account for scr refresh
            cross.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(cross, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'cross.started')
            # update status
            cross.status = STARTED
            cross.setAutoDraw(True)
        
        # if cross is active this frame...
        if cross.status == STARTED:
            # update params
            pass
        
        # if cross is stopping this frame...
        if cross.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > cross.tStartRefresh + breath_duration-frameTolerance:
                # keep track of stop time/frame for later
                cross.tStop = t  # not accounting for scr refresh
                cross.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cross.stopped')
                # update status
                cross.status = FINISHED
                cross.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in _01_08_rest_state_trialComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "_01_08_rest_state_trial" ---
    for thisComponent in _01_08_rest_state_trialComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('_01_08_rest_state_trial.stopped', globalClock.getTime())
    # Run 'End Routine' code from t_stim
    outlet.push_sample(x=[rest_state_counter+9])
    # the Routine "_01_08_rest_state_trial" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "task_end_question" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('task_end_question.started', globalClock.getTime())
    # Run 'Begin Routine' code from code_questionaire
    question_counter += 1
    current_letter = chr(64 + question_counter)  # 65 is ASCII for 'A'
    question_text = 'Questionnaire: ' + current_letter
    
    outlet.push_sample(x=[100+question_counter])
    text_question_number.setText(question_text)
    exp_end_bip.setSound('sound/bip.wav', secs=1.0, hamming=True)
    exp_end_bip.setVolume(1.0, log=False)
    exp_end_bip.seek(0)
    key_q_stress.keys = []
    key_q_stress.rt = []
    _key_q_stress_allKeys = []
    slider.reset()
    slider2.reset()
    # Run 'Begin Routine' code from code
    slider.marker.size = (0.01, 0.01)
    slider2.marker.size = (0.01, 0.01)
    
    # keep track of which components have finished
    task_end_questionComponents = [text_question_number, text_q_stress, exp_end_bip, key_q_stress, slider, slider2]
    for thisComponent in task_end_questionComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "task_end_question" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_question_number* updates
        
        # if text_question_number is starting this frame...
        if text_question_number.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_question_number.frameNStart = frameN  # exact frame index
            text_question_number.tStart = t  # local t and not account for scr refresh
            text_question_number.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_question_number, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_question_number.started')
            # update status
            text_question_number.status = STARTED
            text_question_number.setAutoDraw(True)
        
        # if text_question_number is active this frame...
        if text_question_number.status == STARTED:
            # update params
            pass
        
        # *text_q_stress* updates
        
        # if text_q_stress is starting this frame...
        if text_q_stress.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_q_stress.frameNStart = frameN  # exact frame index
            text_q_stress.tStart = t  # local t and not account for scr refresh
            text_q_stress.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_q_stress, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_q_stress.started')
            # update status
            text_q_stress.status = STARTED
            text_q_stress.setAutoDraw(True)
        
        # if text_q_stress is active this frame...
        if text_q_stress.status == STARTED:
            # update params
            pass
        
        # if exp_end_bip is starting this frame...
        if exp_end_bip.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            exp_end_bip.frameNStart = frameN  # exact frame index
            exp_end_bip.tStart = t  # local t and not account for scr refresh
            exp_end_bip.tStartRefresh = tThisFlipGlobal  # on global time
            # add timestamp to datafile
            thisExp.addData('exp_end_bip.started', tThisFlipGlobal)
            # update status
            exp_end_bip.status = STARTED
            exp_end_bip.play(when=win)  # sync with win flip
        
        # if exp_end_bip is stopping this frame...
        if exp_end_bip.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > exp_end_bip.tStartRefresh + 1.0-frameTolerance:
                # keep track of stop time/frame for later
                exp_end_bip.tStop = t  # not accounting for scr refresh
                exp_end_bip.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'exp_end_bip.stopped')
                # update status
                exp_end_bip.status = FINISHED
                exp_end_bip.stop()
        # update exp_end_bip status according to whether it's playing
        if exp_end_bip.isPlaying:
            exp_end_bip.status = STARTED
        elif exp_end_bip.isFinished:
            exp_end_bip.status = FINISHED
        
        # *key_q_stress* updates
        waitOnFlip = False
        
        # if key_q_stress is starting this frame...
        if key_q_stress.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_q_stress.frameNStart = frameN  # exact frame index
            key_q_stress.tStart = t  # local t and not account for scr refresh
            key_q_stress.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_q_stress, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_q_stress.started')
            # update status
            key_q_stress.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_q_stress.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_q_stress.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_q_stress.status == STARTED and not waitOnFlip:
            theseKeys = key_q_stress.getKeys(keyList=['f1'], ignoreKeys=["escape"], waitRelease=False)
            _key_q_stress_allKeys.extend(theseKeys)
            if len(_key_q_stress_allKeys):
                key_q_stress.keys = _key_q_stress_allKeys[-1].name  # just the last key pressed
                key_q_stress.rt = _key_q_stress_allKeys[-1].rt
                key_q_stress.duration = _key_q_stress_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *slider* updates
        
        # if slider is starting this frame...
        if slider.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            slider.frameNStart = frameN  # exact frame index
            slider.tStart = t  # local t and not account for scr refresh
            slider.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(slider, 'tStartRefresh')  # time at next scr refresh
            # update status
            slider.status = STARTED
            slider.setAutoDraw(True)
        
        # if slider is active this frame...
        if slider.status == STARTED:
            # update params
            pass
        
        # Check slider for response to end Routine
        if slider.getRating() is not None and slider.status == STARTED:
            continueRoutine = False
        
        # *slider2* updates
        
        # if slider2 is starting this frame...
        if slider2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            slider2.frameNStart = frameN  # exact frame index
            slider2.tStart = t  # local t and not account for scr refresh
            slider2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(slider2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'slider2.started')
            # update status
            slider2.status = STARTED
            slider2.setAutoDraw(True)
        
        # if slider2 is active this frame...
        if slider2.status == STARTED:
            # update params
            pass
        
        # Check slider2 for response to end Routine
        if slider2.getRating() is not None and slider2.status == STARTED:
            continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in task_end_questionComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "task_end_question" ---
    for thisComponent in task_end_questionComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('task_end_question.stopped', globalClock.getTime())
    # Run 'End Routine' code from code_questionaire
    outlet.push_sample(x=[199])
    exp_end_bip.pause()  # ensure sound has stopped at end of Routine
    # check responses
    if key_q_stress.keys in ['', [], None]:  # No response was made
        key_q_stress.keys = None
    thisExp.addData('key_q_stress.keys',key_q_stress.keys)
    if key_q_stress.keys != None:  # we had a response
        thisExp.addData('key_q_stress.rt', key_q_stress.rt)
        thisExp.addData('key_q_stress.duration', key_q_stress.duration)
    thisExp.nextEntry()
    thisExp.addData('slider2.response', slider2.getRating())
    thisExp.addData('slider2.rt', slider2.getRT())
    # the Routine "task_end_question" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "End" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('End.started', globalClock.getTime())
    # Run 'Begin Routine' code from t_end
    outlet.push_sample(x=[9])
    
    #Clean up
    import gc
    gc.collect()  # Force garbage collection
    
    # Just clear the screen and wait
    win.flip()
    core.wait(2.0)
    thankyou.setSound('sound/thankyou.wav', secs=1.0, hamming=True)
    thankyou.setVolume(1.0, log=False)
    thankyou.seek(0)
    # keep track of which components have finished
    EndComponents = [EndMessage, thankyou]
    for thisComponent in EndComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "End" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 1.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *EndMessage* updates
        
        # if EndMessage is starting this frame...
        if EndMessage.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            EndMessage.frameNStart = frameN  # exact frame index
            EndMessage.tStart = t  # local t and not account for scr refresh
            EndMessage.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(EndMessage, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'EndMessage.started')
            # update status
            EndMessage.status = STARTED
            EndMessage.setAutoDraw(True)
        
        # if EndMessage is active this frame...
        if EndMessage.status == STARTED:
            # update params
            pass
        
        # if EndMessage is stopping this frame...
        if EndMessage.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > EndMessage.tStartRefresh + 1.0-frameTolerance:
                # keep track of stop time/frame for later
                EndMessage.tStop = t  # not accounting for scr refresh
                EndMessage.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'EndMessage.stopped')
                # update status
                EndMessage.status = FINISHED
                EndMessage.setAutoDraw(False)
        
        # if thankyou is starting this frame...
        if thankyou.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            thankyou.frameNStart = frameN  # exact frame index
            thankyou.tStart = t  # local t and not account for scr refresh
            thankyou.tStartRefresh = tThisFlipGlobal  # on global time
            # add timestamp to datafile
            thisExp.addData('thankyou.started', tThisFlipGlobal)
            # update status
            thankyou.status = STARTED
            thankyou.play(when=win)  # sync with win flip
        
        # if thankyou is stopping this frame...
        if thankyou.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > thankyou.tStartRefresh + 1.0-frameTolerance:
                # keep track of stop time/frame for later
                thankyou.tStop = t  # not accounting for scr refresh
                thankyou.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'thankyou.stopped')
                # update status
                thankyou.status = FINISHED
                thankyou.stop()
        # update thankyou status according to whether it's playing
        if thankyou.isPlaying:
            thankyou.status = STARTED
        elif thankyou.isFinished:
            thankyou.status = FINISHED
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in EndComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "End" ---
    for thisComponent in EndComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('End.stopped', globalClock.getTime())
    thankyou.pause()  # ensure sound has stopped at end of Routine
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-1.000000)
    
    
    
    # mark experiment as finished
    endExperiment(thisExp, win=win, inputs=inputs)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, inputs=None, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # shut down eyetracker, if there is one
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
    logging.flush()


def quit(thisExp, win=None, inputs=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    inputs : dict
        Dictionary of input devices by name.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    inputs = setupInputs(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win, 
        inputs=inputs
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win, inputs=inputs)
