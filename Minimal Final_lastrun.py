#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.3),
    on November 01, 2024, at 18:40
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
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
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.2.3'
expName = 'Minimal Final'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1920, 1080]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
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
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
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
        originPath='C:\\Users\\hourm\\Dropbox\\Projects\\PhD\\T Lab\\Minimal task final\\Minimal Final_lastrun.py',
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
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
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
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=1,
            winType='pyglet', allowGUI=True, allowStencil=False,
            monitor='default', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units=None,
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = None
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
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
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    # Setup iohub experiment
    ioConfig['Experiment'] = dict(filename=thisExp.dataFileName)
    
    # Start ioHub server
    ioServer = io.launchHubServer(window=win, **ioConfig)
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    if deviceManager.getDevice('key_resp') is None:
        # initialise key_resp
        key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
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
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
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
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
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
    
    # --- Initialize components for Routine "text" ---
    Intro = visual.TextStim(win=win, name='Intro',
        text='Press space to start',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp = keyboard.Keyboard(deviceName='key_resp')
    
    # --- Initialize components for Routine "Stimuli" ---
    # Run 'Begin Experiment' code from code_2
    import random
    
    # Define the possible options
    shapes = ["square", "triangle", "circle"]
    colors = ["green", "red", "blue"]
    quantities = [1, 2, 3]
    
    # Declare global variables
    chosen_shape = None
    chosen_color = None
    chosen_quantity = None
    
    # List to collect data
    data_collection = []
    
    # Define a function to create a shape
    def create_shape(shape, color, position):
        if shape == "square":
            return visual.Rect(win, width=0.6, height=0.6, fillColor=color, pos=position)
        elif shape == "triangle":
            return visual.ShapeStim(win, vertices=[(-0.3, -0.3), (0.3, -0.3), (0, 0.3)], fillColor=color, pos=position)
        elif shape == "circle":
            return visual.Circle(win, radius=0.3, fillColor=color, pos=position)
    
    # Generate positions for the stimuli
    positions = [(0, 0), (-3, 0), (3, 0)]
    
    
    
    
    
    # --- Initialize components for Routine "mask" ---
    image = visual.ImageStim(
        win=win,
        name='image', 
        image='C:/Users/hourm/Dropbox/Projects/PhD/T Lab/Stimulu.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    
    # --- Initialize components for Routine "Color" ---
    outer_circle = visual.ShapeStim(
        win=win, name='outer_circle',
        size=(0.6), vertices='circle',
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[-1.0000, 0.0039, -1.0000], fillColor=[-1.0000, 0.0039, -1.0000],
        opacity=None, depth=0.0, interpolate=True)
    middle_circle = visual.ShapeStim(
        win=win, name='middle_circle',
        size=(0.4), vertices='circle',
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, 1.0000], fillColor=[-1.0000, -1.0000, 1.0000],
        opacity=None, depth=-1.0, interpolate=True)
    inner_circle = visual.ShapeStim(
        win=win, name='inner_circle',
        size=(0.2), vertices='circle',
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[1.0000, -1.0000, -1.0000], fillColor=[1.0000, -1.0000, -1.0000],
        opacity=None, depth=-2.0, interpolate=True)
    v_line = visual.Line(
        win=win, name='v_line',
        size=(0.01, 2),
        ori=0.0, pos=(0,0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=None, depth=-3.0, interpolate=True)
    h_line = visual.ShapeStim(
        win=win, name='h_line',
        size=(2, 0.01), vertices='triangle',
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=None, depth=-4.0, interpolate=True)
    mouse = event.Mouse(win=win)
    x, y = [None, None]
    mouse.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "ShapeChoice" ---
    Triangle = visual.ShapeStim(
        win=win, name='Triangle',
        size=(0.3, 0.3), vertices='triangle',
        ori=0.0, pos=(0.5, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    Circle = visual.ShapeStim(
        win=win, name='Circle',
        size=(0.3, 0.3), vertices='circle',
        ori=0.0, pos=(-0.5, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    Square = visual.Rect(
        win=win, name='Square',
        width=(0.3, 0.3)[0], height=(0.3, 0.3)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[1.0000, 1.0000, 1.0000], fillColor=[1.0000, 1.0000, 1.0000],
        opacity=None, depth=-2.0, interpolate=True)
    mouse_resp = event.Mouse(win=win)
    x, y = [None, None]
    mouse_resp.mouseClock = core.Clock()
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "text" ---
    # create an object to store info about Routine text
    text = data.Routine(
        name='text',
        components=[Intro, key_resp],
    )
    text.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp
    key_resp.keys = []
    key_resp.rt = []
    _key_resp_allKeys = []
    # store start times for text
    text.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    text.tStart = globalClock.getTime(format='float')
    text.status = STARTED
    thisExp.addData('text.started', text.tStart)
    text.maxDuration = None
    # keep track of which components have finished
    textComponents = text.components
    for thisComponent in text.components:
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
    
    # --- Run Routine "text" ---
    text.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *Intro* updates
        
        # if Intro is starting this frame...
        if Intro.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Intro.frameNStart = frameN  # exact frame index
            Intro.tStart = t  # local t and not account for scr refresh
            Intro.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Intro, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Intro.started')
            # update status
            Intro.status = STARTED
            Intro.setAutoDraw(True)
        
        # if Intro is active this frame...
        if Intro.status == STARTED:
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
            theseKeys = key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
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
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            text.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in text.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "text" ---
    for thisComponent in text.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for text
    text.tStop = globalClock.getTime(format='float')
    text.tStopRefresh = tThisFlipGlobal
    thisExp.addData('text.stopped', text.tStop)
    # check responses
    if key_resp.keys in ['', [], None]:  # No response was made
        key_resp.keys = None
    thisExp.addData('key_resp.keys',key_resp.keys)
    if key_resp.keys != None:  # we had a response
        thisExp.addData('key_resp.rt', key_resp.rt)
        thisExp.addData('key_resp.duration', key_resp.duration)
    thisExp.nextEntry()
    # the Routine "text" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "Stimuli" ---
    # create an object to store info about Routine Stimuli
    Stimuli = data.Routine(
        name='Stimuli',
        components=[],
    )
    Stimuli.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from code_2
    import random
    
    # Randomly select shape, color, and quantity for each trial
    chosen_shape = random.choice(shapes)
    chosen_color = random.choice(colors)
    chosen_quantity = random.choice(quantities)
    
    # Generate fixed x positions based on the quantity
    stimuli = []
    positions = []  # List to store fixed positions
    fixed_y = 0  # Fixed y-coordinate for all shapes
    
    # Define positions based on the quantity
    if chosen_quantity == 1:
        positions = [(0, fixed_y)]
    elif chosen_quantity == 2:
        spacing = 0.3  # Adjust spacing as needed
        positions = [(-spacing, fixed_y), (spacing, fixed_y)]
    elif chosen_quantity == 3:
        spacing = 0.9  # Adjust spacing as needed
        positions = [(-spacing, fixed_y), (0, fixed_y), (spacing, fixed_y)]
    
    # Create and store the shapes at these fixed positions
    for i in range(chosen_quantity):
        shape = create_shape(chosen_shape, chosen_color, positions[i])
        stimuli.append(shape)
    
    # Draw all shapes
    for stimulus in stimuli:
        stimulus.draw()
    
    win.flip()  # Update the display
    core.wait(0.5)  # Wait for 500ms
    
    # store start times for Stimuli
    Stimuli.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Stimuli.tStart = globalClock.getTime(format='float')
    Stimuli.status = STARTED
    thisExp.addData('Stimuli.started', Stimuli.tStart)
    Stimuli.maxDuration = None
    # keep track of which components have finished
    StimuliComponents = Stimuli.components
    for thisComponent in Stimuli.components:
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
    
    # --- Run Routine "Stimuli" ---
    Stimuli.forceEnded = routineForceEnded = not continueRoutine
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
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            Stimuli.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Stimuli.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Stimuli" ---
    for thisComponent in Stimuli.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Stimuli
    Stimuli.tStop = globalClock.getTime(format='float')
    Stimuli.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Stimuli.stopped', Stimuli.tStop)
    # Run 'End Routine' code from code_2
    # Record the variables in the Psychopy data file
    
    thisExp.saveAsWideText('data/output_data.csv')
    thisExp.addData('shape', chosen_shape)
    thisExp.addData('color', chosen_color)
    thisExp.addData('quantity', chosen_quantity)
    
    
    thisExp.nextEntry()
    # the Routine "Stimuli" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "mask" ---
    # create an object to store info about Routine mask
    mask = data.Routine(
        name='mask',
        components=[image],
    )
    mask.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for mask
    mask.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    mask.tStart = globalClock.getTime(format='float')
    mask.status = STARTED
    thisExp.addData('mask.started', mask.tStart)
    mask.maxDuration = None
    # keep track of which components have finished
    maskComponents = mask.components
    for thisComponent in mask.components:
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
    
    # --- Run Routine "mask" ---
    mask.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 1.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *image* updates
        
        # if image is starting this frame...
        if image.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            image.frameNStart = frameN  # exact frame index
            image.tStart = t  # local t and not account for scr refresh
            image.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(image, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'image.started')
            # update status
            image.status = STARTED
            image.setAutoDraw(True)
        
        # if image is active this frame...
        if image.status == STARTED:
            # update params
            pass
        
        # if image is stopping this frame...
        if image.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > image.tStartRefresh + 0.5-frameTolerance:
                # keep track of stop time/frame for later
                image.tStop = t  # not accounting for scr refresh
                image.tStopRefresh = tThisFlipGlobal  # on global time
                image.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image.stopped')
                # update status
                image.status = FINISHED
                image.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            mask.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in mask.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "mask" ---
    for thisComponent in mask.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for mask
    mask.tStop = globalClock.getTime(format='float')
    mask.tStopRefresh = tThisFlipGlobal
    thisExp.addData('mask.stopped', mask.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if mask.maxDurationReached:
        routineTimer.addTime(-mask.maxDuration)
    elif mask.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-1.000000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "Color" ---
    # create an object to store info about Routine Color
    Color = data.Routine(
        name='Color',
        components=[outer_circle, middle_circle, inner_circle, v_line, h_line, mouse],
    )
    Color.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from code
    # Begin Routine tab
    continueRoutine = True
    response_counter = 0  # Ensure response counter resets each time
    
    # setup some python lists for storing info about the mouse
    mouse.x = []
    mouse.y = []
    mouse.leftButton = []
    mouse.midButton = []
    mouse.rightButton = []
    mouse.time = []
    mouse.clicked_outer_circle = []
    mouse.clicked_middle_circle = []
    mouse.clicked_inner_circle = []
    gotValidClick = False  # until a click is received
    # store start times for Color
    Color.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Color.tStart = globalClock.getTime(format='float')
    Color.status = STARTED
    thisExp.addData('Color.started', Color.tStart)
    Color.maxDuration = None
    # keep track of which components have finished
    ColorComponents = Color.components
    for thisComponent in Color.components:
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
    
    # --- Run Routine "Color" ---
    Color.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *outer_circle* updates
        
        # if outer_circle is starting this frame...
        if outer_circle.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            outer_circle.frameNStart = frameN  # exact frame index
            outer_circle.tStart = t  # local t and not account for scr refresh
            outer_circle.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(outer_circle, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'outer_circle.started')
            # update status
            outer_circle.status = STARTED
            outer_circle.setAutoDraw(True)
        
        # if outer_circle is active this frame...
        if outer_circle.status == STARTED:
            # update params
            pass
        
        # *middle_circle* updates
        
        # if middle_circle is starting this frame...
        if middle_circle.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            middle_circle.frameNStart = frameN  # exact frame index
            middle_circle.tStart = t  # local t and not account for scr refresh
            middle_circle.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(middle_circle, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'middle_circle.started')
            # update status
            middle_circle.status = STARTED
            middle_circle.setAutoDraw(True)
        
        # if middle_circle is active this frame...
        if middle_circle.status == STARTED:
            # update params
            pass
        
        # *inner_circle* updates
        
        # if inner_circle is starting this frame...
        if inner_circle.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            inner_circle.frameNStart = frameN  # exact frame index
            inner_circle.tStart = t  # local t and not account for scr refresh
            inner_circle.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(inner_circle, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'inner_circle.started')
            # update status
            inner_circle.status = STARTED
            inner_circle.setAutoDraw(True)
        
        # if inner_circle is active this frame...
        if inner_circle.status == STARTED:
            # update params
            pass
        
        # *v_line* updates
        
        # if v_line is starting this frame...
        if v_line.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            v_line.frameNStart = frameN  # exact frame index
            v_line.tStart = t  # local t and not account for scr refresh
            v_line.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(v_line, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'v_line.started')
            # update status
            v_line.status = STARTED
            v_line.setAutoDraw(True)
        
        # if v_line is active this frame...
        if v_line.status == STARTED:
            # update params
            pass
        
        # *h_line* updates
        
        # if h_line is starting this frame...
        if h_line.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            h_line.frameNStart = frameN  # exact frame index
            h_line.tStart = t  # local t and not account for scr refresh
            h_line.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(h_line, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'h_line.started')
            # update status
            h_line.status = STARTED
            h_line.setAutoDraw(True)
        
        # if h_line is active this frame...
        if h_line.status == STARTED:
            # update params
            pass
        # Run 'Each Frame' code from code
        import math
        
        # Begin Routine tab
        response_counter = 0  # Ensure response counter resets each time
        max_responses = 1     # Set to 1 if you want to end after one click
        
        
        # Check the mouse button state each frame
        mouse_buttons = mouse.getPressed()
        
        # Only proceed if left mouse button is pressed and we haven't reached max responses
        if mouse_buttons[0] and response_counter < max_responses:  # Checks if the left button is pressed
            # Check if the mouse clicked on any of the circles and set the region and color
            if mouse.isPressedIn(inner_circle):
                region = "inner region"
                color = "Red"
            elif mouse.isPressedIn(middle_circle):
                region = "middle region"
                color = "Blue"
            elif mouse.isPressedIn(outer_circle):
                region = "outer region"
                color = "Green"
            else:
                region = "outside the circles"
                color = "None"
            
            # Add region and color to data
            thisExp.addData("region", region)
            thisExp.addData("color", color)
        
            # Get the mouse position
            mouse_pos = mouse.getPos()
            radius = (mouse_pos[0]**2 + mouse_pos[1]**2)**0.5
            
            # Add radius to data
            thisExp.addData("radius", radius)
        
            # Calculate the angle of the click
            angle = math.degrees(math.atan2(mouse_pos[1], mouse_pos[0]))
            if angle < 0:
                angle += 360  # Normalize angle to be between 0 and 360 degrees
        
            # Add angle to data
            thisExp.addData("angle", angle)
        
            # Assign quadrant numbers based on 90-degree angle ranges
            if 0 <= angle < 90:
                quadrant = 1
            elif 90 <= angle < 180:
                quadrant = 2
            elif 180 <= angle < 270:
                quadrant = 3
            elif 270 <= angle < 360:
                quadrant = 4
            else:
                quadrant = "Unknown Quadrant"
            
            # Add quadrant to data
            thisExp.addData("quadrant", quadrant)
        
            # Increment the response counter
            response_counter += 1
        
            # End routine if max responses reached
            if response_counter >= max_responses:
                continueRoutine = False  # Only set this here
                print("Routine ending: Max responses reached")  # Debugging line
        
        
        # *mouse* updates
        
        # if mouse is starting this frame...
        if mouse.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            mouse.frameNStart = frameN  # exact frame index
            mouse.tStart = t  # local t and not account for scr refresh
            mouse.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(mouse, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'mouse.started')
            # update status
            mouse.status = STARTED
            mouse.mouseClock.reset()
            prevButtonState = [0, 0, 0]  # if now button is down we will treat as 'new' click
        
        # if mouse is stopping this frame...
        if mouse.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > mouse.tStartRefresh + 0-frameTolerance:
                # keep track of stop time/frame for later
                mouse.tStop = t  # not accounting for scr refresh
                mouse.tStopRefresh = tThisFlipGlobal  # on global time
                mouse.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'mouse.stopped')
                # update status
                mouse.status = FINISHED
        if mouse.status == STARTED:  # only update if started and not finished!
            buttons = mouse.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    # check if the mouse was inside our 'clickable' objects
                    gotValidClick = False
                    clickableList = environmenttools.getFromNames([outer_circle , middle_circle, inner_circle], namespace=locals())
                    for obj in clickableList:
                        # is this object clicked on?
                        if obj.contains(mouse):
                            gotValidClick = True
                            mouse.clicked_outer_circle.append(obj.outer_circle)
                            mouse.clicked_middle_circle.append(obj.middle_circle)
                            mouse.clicked_inner_circle.append(obj.inner_circle)
                    if not gotValidClick:
                        mouse.clicked_outer_circle.append(None)
                        mouse.clicked_middle_circle.append(None)
                        mouse.clicked_inner_circle.append(None)
                    x, y = mouse.getPos()
                    mouse.x.append(x)
                    mouse.y.append(y)
                    buttons = mouse.getPressed()
                    mouse.leftButton.append(buttons[0])
                    mouse.midButton.append(buttons[1])
                    mouse.rightButton.append(buttons[2])
                    mouse.time.append(mouse.mouseClock.getTime())
                    if gotValidClick:
                        continueRoutine = False  # end routine on response
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            Color.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Color.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Color" ---
    for thisComponent in Color.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Color
    Color.tStop = globalClock.getTime(format='float')
    Color.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Color.stopped', Color.tStop)
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('mouse.x', mouse.x)
    thisExp.addData('mouse.y', mouse.y)
    thisExp.addData('mouse.leftButton', mouse.leftButton)
    thisExp.addData('mouse.midButton', mouse.midButton)
    thisExp.addData('mouse.rightButton', mouse.rightButton)
    thisExp.addData('mouse.time', mouse.time)
    thisExp.addData('mouse.clicked_outer_circle', mouse.clicked_outer_circle)
    thisExp.addData('mouse.clicked_middle_circle', mouse.clicked_middle_circle)
    thisExp.addData('mouse.clicked_inner_circle', mouse.clicked_inner_circle)
    thisExp.nextEntry()
    # the Routine "Color" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "ShapeChoice" ---
    # create an object to store info about Routine ShapeChoice
    ShapeChoice = data.Routine(
        name='ShapeChoice',
        components=[Triangle, Circle, Square, mouse_resp],
    )
    ShapeChoice.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # setup some python lists for storing info about the mouse_resp
    mouse_resp.x = []
    mouse_resp.y = []
    mouse_resp.leftButton = []
    mouse_resp.midButton = []
    mouse_resp.rightButton = []
    mouse_resp.time = []
    mouse_resp.clicked_name = []
    gotValidClick = False  # until a click is received
    # store start times for ShapeChoice
    ShapeChoice.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    ShapeChoice.tStart = globalClock.getTime(format='float')
    ShapeChoice.status = STARTED
    thisExp.addData('ShapeChoice.started', ShapeChoice.tStart)
    ShapeChoice.maxDuration = None
    # keep track of which components have finished
    ShapeChoiceComponents = ShapeChoice.components
    for thisComponent in ShapeChoice.components:
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
    
    # --- Run Routine "ShapeChoice" ---
    ShapeChoice.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *Triangle* updates
        
        # if Triangle is starting this frame...
        if Triangle.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Triangle.frameNStart = frameN  # exact frame index
            Triangle.tStart = t  # local t and not account for scr refresh
            Triangle.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Triangle, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Triangle.started')
            # update status
            Triangle.status = STARTED
            Triangle.setAutoDraw(True)
        
        # if Triangle is active this frame...
        if Triangle.status == STARTED:
            # update params
            pass
        
        # *Circle* updates
        
        # if Circle is starting this frame...
        if Circle.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Circle.frameNStart = frameN  # exact frame index
            Circle.tStart = t  # local t and not account for scr refresh
            Circle.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Circle, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Circle.started')
            # update status
            Circle.status = STARTED
            Circle.setAutoDraw(True)
        
        # if Circle is active this frame...
        if Circle.status == STARTED:
            # update params
            pass
        
        # *Square* updates
        
        # if Square is starting this frame...
        if Square.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Square.frameNStart = frameN  # exact frame index
            Square.tStart = t  # local t and not account for scr refresh
            Square.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Square, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Square.started')
            # update status
            Square.status = STARTED
            Square.setAutoDraw(True)
        
        # if Square is active this frame...
        if Square.status == STARTED:
            # update params
            pass
        # *mouse_resp* updates
        
        # if mouse_resp is starting this frame...
        if mouse_resp.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            mouse_resp.frameNStart = frameN  # exact frame index
            mouse_resp.tStart = t  # local t and not account for scr refresh
            mouse_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(mouse_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.addData('mouse_resp.started', t)
            # update status
            mouse_resp.status = STARTED
            mouse_resp.mouseClock.reset()
            prevButtonState = mouse_resp.getPressed()  # if button is down already this ISN'T a new click
        if mouse_resp.status == STARTED:  # only update if started and not finished!
            x, y = mouse_resp.getPos()
            mouse_resp.x.append(x)
            mouse_resp.y.append(y)
            buttons = mouse_resp.getPressed()
            mouse_resp.leftButton.append(buttons[0])
            mouse_resp.midButton.append(buttons[1])
            mouse_resp.rightButton.append(buttons[2])
            mouse_resp.time.append(mouse_resp.mouseClock.getTime())
            buttons = mouse_resp.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    # check if the mouse was inside our 'clickable' objects
                    gotValidClick = False
                    clickableList = environmenttools.getFromNames([Triangle , Circle, Square], namespace=locals())
                    for obj in clickableList:
                        # is this object clicked on?
                        if obj.contains(mouse_resp):
                            gotValidClick = True
                            mouse_resp.clicked_name.append(obj.name)
                    if not gotValidClick:
                        mouse_resp.clicked_name.append(None)
                    if gotValidClick:
                        continueRoutine = False  # end routine on response
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            ShapeChoice.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in ShapeChoice.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "ShapeChoice" ---
    for thisComponent in ShapeChoice.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for ShapeChoice
    ShapeChoice.tStop = globalClock.getTime(format='float')
    ShapeChoice.tStopRefresh = tThisFlipGlobal
    thisExp.addData('ShapeChoice.stopped', ShapeChoice.tStop)
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('mouse_resp.x', mouse_resp.x)
    thisExp.addData('mouse_resp.y', mouse_resp.y)
    thisExp.addData('mouse_resp.leftButton', mouse_resp.leftButton)
    thisExp.addData('mouse_resp.midButton', mouse_resp.midButton)
    thisExp.addData('mouse_resp.rightButton', mouse_resp.rightButton)
    thisExp.addData('mouse_resp.time', mouse_resp.time)
    thisExp.addData('mouse_resp.clicked_name', mouse_resp.clicked_name)
    thisExp.nextEntry()
    # the Routine "ShapeChoice" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


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


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
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
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
