';/'"""
 Audio Processing
"""

import wave
import struct

# No additional imports allowed!


def backwards(sound):
    """ "Takes the mono sound and returns a new mono sound that is
    reversed while retaining the original mono sound
    inputs:
          sound: dictionary, with two key/value pairs:
                rate: sampling rate as an (int)
                samples: list,containing samples where each sample is a float
    returns:
            new reversed mono sound"""
    new_sound={}
    for key,val in sound.items():
        new_sound[key]=val
    new_list=new_sound["samples"]
    new_sound=new_list.reverse()
    return new_sound
    
    # raise NotImplementedError


def mix(sound1, sound2, p):
    """ "Mixes two sounds together to create a new sound.The resulting sound
    takes p times the samples in the first samples in the sound and
    1-p times the samples in the second sound and adds them together

    If the sounds have different sampling rates , returns None
    If the sounds have differet durations return the minimum length of the two sounds.

    inputs:
          sound 1; dictionary
          sound 2:dictionary
          p;float between 0 and 1
    returns:
           newsound:dictionary"""

    # mix 2 good sounds
    if (
        "rate" in sound1.keys()
        and "rate" in sound2.keys()
        and sound1["rate"] != sound2["rate"]
    ):
        print("no")
        return None

    rate = sound1["rate"]  # get rate
    sound1 = sound1["samples"]
    sound2 = sound2["samples"]
    if len(sound1) != len(sound2):
        length = min(len(sound1), len(sound2))
    elif len(sound1) == len(sound2):
        length = len(sound1)
    else:
        print("whoops")
        return

    samples = []
    x = 0
    while x <= length:
        s2, s1 = p * sound1[x], sound2[x] * (1 - p)
        samples.append(s1 + s2)  # add sounds
        x += 1
        if x == length:  # end
            break

    return {"rate": rate, "samples": samples}  # return new sound


def convolve(sound, kernel):
    """
    Applies a filter to a sound, resulting in a new sound that is longer than
    the original mono sound by the length of the kernel - 1.
    Does not modify inputs.
    Args:
        sound: A mono sound dictionary with two key/value pairs:
            * "rate": an int representing the sampling rate, samples per second
            * "samples": a list of floats containing the sampled values
        kernel: A list of numbers
    Returns:
        A new mono sound dictionary.
    """
    samples = []  # a list of scaled sample lists
    for i, scale in enumerate(kernel):
        scaled_sample = [0] * i  # offset scaled sound by filter index
        for x in sound["samples"]:
            scaled_sample.append(scale * x)
        samples.append(scaled_sample)

    # combine samples into one list
    final_sample = [0] * (len(sound["samples"]) + len(kernel) - 1)
    for i, val in enumerate(scaled_sample):
        add_list = []
        for sample in samples:
            # if not long enough, add [0] to end
            while len(sample) < len(final_sample):
                sample += [val]
            add_list.append(sample[i])
        # update final sample with new sample value
        final_sample[i] = sum(add_list)

    return {"rate": sound["rate"], "samples": final_sample}


def echo(sound, num_echoes, delay, scale):
    """
    Compute a new signal consisting of several
    scaled-down and delayed versions
    of the input sound. Does not modify input sound.
    Args:
        sound: a dictionary representing the original mono sound
        num_echoes: int, the number of additional copies of the sound to add
        delay: float, the amount of seconds each echo should be delayed
        scale: float, the amount by which each echo's samples should be scaled
    Returns:
        A new mono sound dictionary resulting from applying the echo effect.
    """
    sample_delay = round(delay * sound["rate"])
    sound_len = len(sound["samples"])
    total_len = sample_delay * num_echoes + sound_len
    echo_sound = {"rate": sound["rate"], "samples": [0] * total_len}
    for i in range(sound_len):
        echo_sound["samples"][i] = sound["samples"][i]
    for i in range(1, num_echoes + 1):
        index = 0
        for j in range(i * sample_delay, i * sample_delay + sound_len):
            echo_sound["samples"][j] += sound["samples"][index] * (scale**i)
            index += 1
    return echo_sound


def pan(sound):
    """ "Creates a neat spatial sound effect
    In particular, if our sound is
        We scale the first sample in the right channel by 0,
           the second by 1/N-1,the third by 2/N-1 and the last by 1
        At the same time, we scale the first sample in the
         left channel by 1,the second by 1-1/N-1 and the last by 0
    Args:
        sound: a dictionary representing the original mono sound
    Returns:
        a new sound dictionary"""
    newsound = {}
    for keys, values in sound.items():
        # print(len(sound["left"]))
        if keys == "rate":
            newsound["rate"] = values
        elif keys == "left":
            newvalues = []
            for i,j in enumerate(values):
                newvalues.append((1 - (i / (len(values) - 1))) * values[i])
            newsound["left"] = newvalues
        elif keys == "right":
            newvalues = []
            for i,j in enumerate(values):
                newvalues.append((i / (len(values) - 1)) * values[i])
            newsound["right"] = newvalues
    return newsound

    # raise NotImplementedError


def remove_vocals(sound):
    """''removing vocals from a piece of music,
    creating a version of the song that would be appropriate as
    a backing track for karaoke night.
    Args:
        This effect will take a stereo sound as input,

    Results:

         a mono sound as output.
    """
    newsound = {}
    newvalues = []
    values_left = []
    values_right = []
    for key, values in sound.items():
        if key == "rate":
            newsound["rate"] = values
        elif key == "left":
            values_left = values
        elif key == "right":
            values_right = values
    for i in range(len(values)):
        result = values_left[i] - values_right[i]
        newvalues.append(result)
    newsound["samples"] = newvalues
    return newsound

    # raise NotImplementedError


# below are helper functions for converting back-and-forth between WAV files
# and our internal dictionary representation for sounds


def bass_boost_kernel(boost, scale=0):
    """
    Constructs a kernel that acts as a bass-boost filter.

    We start by making a low-pass filter, whose frequency response is given by
    (1/2 + 1/2cos(Omega)) ^ N

    Then we scale that piece up and add a copy of the original signal back in.

    Args:
        boost: an int that controls the frequencies that are boosted (0 will
            boost all frequencies roughly equally, and larger values allow more
            focus on the lowest frequencies in the input sound).
        scale: a float, default value of 0 means no boosting at all, and larger
            values boost the low-frequency content more);

    Returns:
        A list of floats representing a bass boost kernel.
    """
    # make this a fake "sound" so that we can use the convolve function
    base = {"rate": 0, "samples": [0.25, 0.5, 0.25]}
    kernel = {"rate": 0, "samples": [0.25, 0.5, 0.25]}
    for i in range(boost):
        kernel = convolve(kernel, base["samples"])
    kernel = kernel["samples"]

    # at this point, the kernel will be acting as a low-pass filter, so we
    # scale up the values by the given scale, and add in a value in the middle
    # to get a (delayed) copy of the original
    kernel = [i * scale for i in kernel]
    kernel[len(kernel) // 2] += 1

    return kernel


def load_wav(filename, stereo=False):
    """
    Load a file and return a sound dictionary.

    Args:
        filename: string ending in '.wav' representing the sound file
        stereo: bool, by default sound is loaded as mono, if True sound will
            have left and right stereo channels.

    Returns:
        A dictionary representing that sound.
    """
    sound_file = wave.open(filename, "r")
    chan, bd, sr, count, _, _ = sound_file.getparams()

    assert bd == 2, "only 16-bit WAV files are supported"

    out = {"rate": sr}

    left = []
    right = []
    for i in range(count):
        frame = sound_file.readframes(1)
        if chan == 2:
            left.append(struct.unpack("<h", frame[:2])[0])
            right.append(struct.unpack("<h", frame[2:])[0])
        else:
            datum = struct.unpack("<h", frame)[0]
            left.append(datum)
            right.append(datum)

    if stereo:
        out["left"] = [i / (2**15) for i in left]
        out["right"] = [i / (2**15) for i in right]
    else:
        samples = [(ls + rs) / 2 for ls, rs in zip(left, right)]
        out["samples"] = [i / (2**15) for i in samples]

    return out


def write_wav(sound, filename):
    """
    Save sound to filename location in a WAV format.

    Args:
        sound: a mono or stereo sound dictionary
        filename: a string ending in .WAV representing the file location to
            save the sound in
    """
    outfile = wave.open(filename, "w")

    if "samples" in sound:
        # mono file
        outfile.setparams((1, 2, sound["rate"], 0, "NONE", "not compressed"))
        out = [int(max(-1, min(1, v)) * (2**15 - 1)) for v in sound["samples"]]
    else:
        # stereo
        outfile.setparams((2, 2, sound["rate"], 0, "NONE", "not compressed"))
        out = []
        for l_val, r_val in zip(sound["left"], sound["right"]):
            l_val = int(max(-1, min(1, l_val)) * (2**15 - 1))
            r_val = int(max(-1, min(1, r_val)) * (2**15 - 1))
            out.append(l_val)
            out.append(r_val)

    outfile.writeframes(b"".join(struct.pack("<h", frame) for frame in out))
    outfile.close()


def case1():
    sound = load_wav("sounds/mystery.wav")
    write_wav(backwards(sound), "mystery_reversed.wav")


def case2():
    sound1 = load_wav("sounds/synth.wav")
    sound2 = load_wav("sounds/water.wav")
    write_wav(mix(sound1, sound2, 0.2), "mixed_synth_water.wav")


def case3():
    sound1 = load_wav("sounds/synth.wav")
    write_wav(echo(sound1, 5, 0.3, 0.6), "echo_chord.wav")


def case4():
    sound = load_wav("sounds/car.wav", stereo=True)
    write_wav(pan(sound), "left_to_right_car.wav")


def case5():
    sound = load_wav("sounds/lookout_mountain.wav", stereo=True)
    write_wav(remove_vocals(sound), "remove_vocals_lookout_mountain.wav")


def case6():
    sound = load_wav("sounds/ice_and_chilli.wav")
    write_wav(
        convolve(sound, bass_boost_kernel(boost=1000, scale=0.5)), "ice_and_chilli.wav"
    )


# 6.101-submit -a audio_processing lab.pyif __name__ == "__main__":
# code in this block will only be run when you explicitly run your script,
# and not when the tests are being run.  this is a good place to put your
# code for generating and saving sounds, or any other code you write for
# testing, etc.

# here is an example of loading a file (note that this is specified as
# sounds/hello.wav, rather than just as hello.wav, to account for the
# sound files being in a different directory than this file)


# write_wav(backwards(hello), "hello_reversed.wav")
