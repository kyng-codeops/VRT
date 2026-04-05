# Example of Hard Telecine mkv

## The test for telecine, interlace, or progressive
```
(vsynth) user@zen24lts:/Volumes/sdata/scratch/movie$ cat ../vapour-scripts/test-idet-full.sh 
ffmpeg -filter:v idet -an -f rawvideo -y /dev/null -i $1 
```

### Running the test command
```
(vsynth) user@zen24lts:/Volumes/sdata/scratch/movie$ ../vapour-scripts/test-idet-full.sh input.mkv
```

The most important part of the output was the bottom including the video stream copied into Google AI Mode to help analyze:

```
  Stream #0:0(eng): Video: rawvideo (I420 / 0x30323449), yuv420p(tv, progressive), 720x480 [SAR 853:720 DAR 853:480], q=2-31, 124291 kb/s, 29.97 fps, 29.97 tbn
    Metadata:
      encoder         : Lavc61.24.100 rawvideo
      BPS-eng         : 6003817
      DURATION-eng    : 02:09:49.281500000
      NUMBER_OF_FRAMES-eng: 186871
      NUMBER_OF_BYTES-eng: 5845678092
      SOURCE_ID-eng   : 0100E0
      _STATISTICS_WRITING_APP-eng: MakeMKV v1.17.8 linux(x64-release)
      _STATISTICS_WRITING_DATE_UTC-eng: 2026-02-26 17:33:56
      _STATISTICS_TAGS-eng: BPS DURATION NUMBER_OF_FRAMES NUMBER_OF_BYTES SOURCE_ID
[vf#0:0 @ 0x566ff7e38c00] More than 1000 frames duplicated23 bitrate=124245.7kbits/s dup=813 drop=0 speed= 270x    
[vf#0:0 @ 0x566ff7e38c00] More than 10000 frames duplicated46 bitrate=124288.2kbits/s dup=9781 drop=0 speed= 182x    
[Parsed_idet_0 @ 0x7cc128002dc0] Repeated Fields: Neither:186813 Top:    28 Bottom:    30up=46045 drop=0 speed= 188x    
[Parsed_idet_0 @ 0x7cc128002dc0] Single frame detection: TFF:   705 BFF:   275 Progressive:130239 Undetermined: 55652
[Parsed_idet_0 @ 0x7cc128002dc0] Multi frame detection: TFF:  3656 BFF:   777 Progressive:182437 Undetermined:     1
[out#0/rawvideo @ 0x566ff7dffb00] video:118181531KiB audio:0KiB subtitle:0KiB other streams:0KiB global headers:0KiB muxing overhead: 0.000000%
frame=233445 fps=5644 q=-0.0 Lsize=118181531KiB time=02:09:49.28 bitrate=124291.7kbits/s dup=46574 drop=0 speed= 188x    
(vsynth) user@zen24lts:/Volumes/sdata/scratch/movie$
```

### Analysis Using Google AI mode LLM:
Your idet results provide a textbook example of a hard-telecined movie source. The key to reading this output is comparing the "Single frame" vs. "Multi frame" detection and looking at the frame counts. 

1. Why it looks "mostly progressive"

    The 3:2 Pattern: In a standard telecine process, film (24 fps) is converted to NTSC (30 fps) by spreading 4 film frames across 5 video frames.

        3 out of 5 frames are physically progressive (fields come from the same moment).
        2 out of 5 frames are interlaced (fields come from different film frames).
    
    Your Data: Your Multi frame detection shows 182,437 Progressive frames out of ~186k total. This high number is normal because idet's multi-frame algorithm is designed to be "sticky"—once it sees a pattern of mostly progressive frames, it tends to classify the whole sequence that way unless there is heavy, constant motion. 

2. The "Hundreds vs. Thousands" discrepancy

    Single Frame (TFF: 705 / BFF: 275): This is the per-frame analysis. These counts represent the specific frames where idet found enough "combing" or "teeth" to be certain they are interlaced.
    Multi Frame (TFF: 3656 / BFF: 777): This is the statistical trend. It shows that while only ~1,000 frames were obviously interlaced to the single-frame detector, the surrounding context suggests over 4,000 frames are part of an interlaced stream.
    High Undetermined (55,652): This usually happens in scenes with low motion (static shots, talking heads). If there’s no movement, there's no "combing" for the filter to detect, so it can't tell if a frame is interlaced or progressive. 

3. The "dup=46574" issue
    
    Your output shows a massive number of duplicated frames (dup=46574). 

    The Cause: Your input is 29.97 fps, but you are piping it to rawvideo at a rate FFmpeg might be trying to "fix" or match to a system clock.

    The Reality: The actual frame count of your movie is 186,871. FFmpeg processed 233,445 frames because it duplicated roughly 1 out of every 4 frames to keep up with the 29.97 fps target while perhaps encountering timestamp gaps or variable frame rate (VFR) metadata. 

Recommendation

The presence of ~4,000+ TFF/BFF frames in a movie-length file confirms Hard Telecine. You should proceed with an Inverse Telecine (IVTC) to get back to the original 23.976 fps film speed.

### Asking Google AI mode for the simplified logic:
#### The strongest indicator isn't mathmatical division of total frames
It is the Frame Rate vs. Content Type:

- Fact A: It is a "Movie" (originally 24fps).
- Fact B: The file is 29.97fps.
- Fact C: idet found any consistent TFF/BFF frames.

If a 24fps movie is inside a 29.97fps container and shows any combing, it is mathematically certain to be a 3:2 pulldown. A truly interlaced (30i) source would show TFF/BFF counts in the tens of thousands, not the low thousands.
#### The "Undetermined" Smoking Gun
The 55,652 Undetermined frames are actually the proof. In a true interlaced video (like a 1990s soap opera), almost every frame has some motion/interlacing. In a telecined movie, the "interlacing" only appears when things move. The high "Undetermined" and "Progressive" counts combined with small bursts of "TFF" confirm that the interlacing is periodic, which is the definition of telecine.
#### The Verdict:
Your idet output is a "textbook" signature of hard-telecined film. The low TFF/BFF count relative to the total frame count simply means your movie has a lot of slow-moving scenes where the interlacing isn't statistically obvious to the filter.


## My Porcessing scripts

### The vapoursynth vpy
```
(vsynth) user@zen24lts:/Volumes/sdata/scratch/movie$ cat vivtc_detel.vpy 
```

```
import vapoursynth as vs
core = vs.core
core.num_threads = 24
core.max_cache_size = 8000

# 1. Source
video = core.ffms2.Source("input.mkv", fpsnum=30000, fpsden=1001)

# 2. IVTC (Inverse Telecine)
# This recovers the original 23.976 fps film frames
video = core.vivtc.VFM(video, order=1) # 1 = Top Field First (standard for DVD)
video = core.vivtc.VDecimate(video)    # Removes the 1 redundant frame in every 5


# 3. Optional keep SAR vs DAR aspect, but prefer phyiscally resize the clean 23.976 fps 
# since PAR often lost between in the vspipe | ffmpeg pipe.
video = core.resize.Spline36(video, width=854, height=480)

video.set_output()
```

### The shell driver script
```
(vsynth) user@zen24lts:/Volumes/sdata/scratch/movie$ cat detel_vspipe-ffmpeg.sh 
```

```
vspipe -c y4m vivtc_detel.vpy - -r 32 | \
ffmpeg -hide_banner -thread_queue_size 4096 \
-itsoffset 0.083 -i - \
  -i "input.mkv" \
  -map 0:v -map 1:a \
  -video_track_timescale 90k \
-c:v hevc_nvenc \
-preset p6 \
-tune hq \
-multipass qres \
-rc vbr -cq 18 -b:v 0 \
-pix_fmt p010le \
-profile:v main10 \
-spatial-aq 1 -temporal-aq 1 -aq-strength 8 \
-rc-lookahead 32 \
  -threads 24 \
  -c:a copy -shortest -movflags +faststart output-ivtc.mp4
```

---
# Google AI Chat reveals TV interlace vs hybrid content

user prompt:

If I said the mkv file were a TV show at 29.97 we know it must be telecine, interlace, or hybrid where some scenes are telecine and some are interlaced. Are there known ratios of TFF or BFF vs progressive like 50% we can use?

    
llm response:

You’ve got the logic down. There isn’t a hard "50%" rule in the code, but in practice, if more than 20–30% of frames with motion are flagged as TFF/BFF, you are almost certainly looking at native interlaced (60i) content.

- live sports or news, 90s soap, reality TV, 70s game or variety shows
    - assume interlace
        1. TFF and or BFF nearly equal to progressive meaining every frame or every other frame in the multi-frame parsing
        2. or nearly all TFF or nearly all BFF in either single-frame or multi-frame parsing

- 90s TV with special effects
    - assume hybrid with people scenes often telecine and special affects interlaced
    - use the TIVTC filter in vapoursynth

- 70s TV not game or variety shows
    - assume telecine because those shows were shot on 35mm 24 fps film
    - use ivtc with fieldmatch and decimate (or TFM and TDecimate in VapourSynth)

- 80s TV is tv show title dependant
    - need a google search for the show title to know how it was shot because it was a decade of transition
    - if the title was shot on film it's telecine
    - if the title was shot on tape it's interlace

- if any of them say they are modern HD re-mastered they are likely progressive



## Possible detector algorithm

Use PyAV and PySceneDetect to find frame ranges that define scenes, then analyze the scenes (i.e. frame ranges).


