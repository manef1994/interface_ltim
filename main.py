import tkinter
import idlelib
from idlelib import window
from tkinter import *
from tkinter import ttk
from ecgdetectors import Detectors
import pyedflib
import tkinter as tk
import pyedflib
import matplotlib
from tkinter import filedialog
from PIL import Image, ImageTk
import os
from tkinter import messagebox
from tkinter import Entry
from tkinter import *
# import trimesh
# from hanene1 import crer
from pandas import np
from scipy.signal import butter, iirnotch, filtfilt
from pathlib import Path
from scipy.signal import butter, iirnotch, lfilter, filtfilt
import matplotlib.pyplot as plt
import neurokit2 as nk


class ReconstructuionSegmentation:

    global path_signal, start_entry, end_entry, Fss, path_browse, fs, signal_input, electrode, notch

    def __init__(self, root):

        #global path_signal, start_entry, end_entry, Fss, path_browse, fs, signal_input

        self.root = root
        self.root.title("Manipulation de signal")
        self.root.geometry("1500x800")
        #image_tim = Image.open("E:\Thése\python\\tim.png")
        #test = ImageTk.PhotoImage(image_tim)

        #label1 = Label(image=test)
        #label1.image = test
        #label1.place(x=10, y=80)

        def butter_lowpass(cutoff, fs, order=2):
            nyq = 0.5 * fs
            normal_cutoff = cutoff / nyq
            b, a = butter(order, normal_cutoff, btype='low', analog=False)
            return b, a

        def butter_highpass(cutoff, sample_rate, order=2):
            nyq = 0.5 * sample_rate
            normal_cutoff = cutoff / nyq
            b, a = butter(order, normal_cutoff, btype='high', analog=False)
            return b, a

        def remove_baseline_wander(data, sample_rate, cutoff=0.05):
            return filter_signal(data=data, cutoff=cutoff, sample_rate=sample_rate, filtertype='notch')

        def butter_bandpass(lowcut, highcut, sample_rate, order=2):
            nyq = 0.5 * sample_rate
            low = lowcut / nyq
            high = highcut / nyq
            b, a = butter(order, [low, high], btype='band')
            return b, a

        def filter_signal(data, cutoff, sample_rate, order=2, filtertype='lowpass', return_top=False):
            if filtertype.lower() == 'lowpass':
                b, a = butter_lowpass(cutoff, sample_rate, order=order)
            elif filtertype.lower() == 'highpass':
                b, a = butter_highpass(cutoff, sample_rate, order=order)
            elif filtertype.lower() == 'bandpass':
                assert type(cutoff) == tuple or list or np.array, 'if bandpass filter is specified, \
            cutoff needs to be array or tuple specifying lower and upper bound: [lower, upper].'
                b, a = butter_bandpass(cutoff[0], cutoff[1], sample_rate, order=order)
            elif filtertype.lower() == 'notch':
                b, a = iirnotch(cutoff, Q=0.005, fs=sample_rate)
            else:
                raise ValueError('filtertype: %s is unknown, available are: \
            lowpass, highpass, bandpass, and notch' % filtertype)

            filtered_data = filtfilt(b, a, data)

            if return_top:
                return np.clip(filtered_data, a_min=0, a_max=None)
            else:
                return filtered_data

        def Browse(main_win=None):
            #plt.close(1)
            #plt.close(2)
            #plt.close(3)
            #plt.close(4)
            #plt.close(5)
            #plt.close(6)
            #plt.close(7)
            #plt.close(8)
            #plt.close(9)
            #plt.close(10)

            global path_signal
            global EDF
            global path_browse
            global electrode
            global Fss
            global path_signal, notch

            # ======= open file selector
            sourceFile = tkinter.filedialog.askopenfilename(parent=main_win,initialdir="E:\\data\\siena-scalp-eeg-database-1.0.0\\",title='Please select a directory', )
            path = sourceFile
            path = path.replace('/', '\\')
            # ======= print path

            if path.endswith('.csv') or path.endswith('.edf') or path.endswith('.txt'):
                if path.endswith('.edf'):
                    Fss.config(state='disabled')
                if path.endswith('.csv') or path.endswith('.txt'):
                    Fss.config(state='normal')
                    electrode.config(state='disabled')
                if path.endswith('.csv') or path.endswith('.txt'):
                    Fss.config(state='normal')
                    electrode.config(state='disabled')

                path_signal = str(path)
                path_browse.config(text = path_signal)
                print('path is\t',path_signal)

                print(str(electrode.get()))

                return path_signal

            else:
                messagebox.showinfo("warninig file missing",
                                    "you must select a CSV or EDF file, please recheck the file selected")

        def Read_signal(path):
            global fs
            global Fss
            global signal_input, notch

            if path.endswith('.csv') or path.endswith('.txt'):
                signal_input = np.loadtxt(path)
                fs = int(float(Fss.get()))

                with open(path, "r") as file1:
                    signal_input = [float(i) for line in file1 for i in line.split(',') if i.strip()]

                fs = int(float(Fss.get()))

            elif path.endswith('.edf'):
                print('edf file is detected\t')
                f = pyedflib.EdfReader(path)
                # ==== n is the number of channels
                n = f.signals_in_file
                ###################################
                signal_input = np.zeros((n, f.getNSamples()[0]))
                signal_labels = f.getSignalLabels()
                signal_samples_fr = f.getSampleFrequencies()

                # extracting the channel where there's ECG
                channel = 0
                for x in range(len(signal_labels)):
                    nn = signal_labels[x]
                    if ((nn == str(electrode.get()))):
                        channel = x
                        fs = signal_samples_fr[x]

                signal_input = f.readSignal(channel)
                fs = signal_samples_fr[channel]
                Fss.config(text = fs)
                print("FS out of the loop equals to: ", fs)
                print("ECG channel equals to: ", channel)

            return signal_input

        def Entropie(path_signal):

            print('path in Entropie function :\t',path_signal)
            signal_input = Read_signal(path_signal)
            print('length of the input signal', len(signal_input))
            len_vald = len(signal_input)

            global start, end
            start = int(start_entry.get()) * fs * 60
            end = int(end_entry.get()) * fs * 60

            print('start time = \t', start)
            print('end time = \t\t', ((end/60)/fs))
            print('validation\t',len_vald)

            if (len_vald >= end):

                signal_input = signal_input[start:end]

                print('length of the input signal equals to:\t', len(signal_input))
                signal_output = filter_signal(data=signal_input, cutoff=int(notch.get()), sample_rate=fs, order=2, filtertype="notch")
                signal_output = filter_signal(data=signal_output, cutoff=20, sample_rate=fs, order=2,filtertype="highpass")
                signal_output = filter_signal(data=signal_output, cutoff=10, sample_rate=fs, order=2,filtertype="lowpass")
                try:
                    print(signal_input)
                    Samp_entropy, fuzzy_entropy, shannon_entropy, approximate_entropy, approximate_entropy2, multi_multiscale, approximate_entropy3, approximate_entropy4, approximate_entropy5, spectral = (
                        [] for i in range(10))

                    six_seconds = fs * 6
                    one_minut = fs * 60
                    ###########################################################
                    # == compute the HRV
                    five_min = []
                    HRV = []
                    RRi = []
                    final_peaks = []
                    lengthh = (int(len(signal_output) / fs)) / 60

                    t = 3
                    min = t * 60 * fs
                    lenn = int(len(signal_input) / min)

                    RRi = []
                    condition = True

                    start = 0
                    end = fs * 60

                    while end <= (len(signal_input)):

                        detectors = Detectors(fs)
                        signal = signal_output[start:end]
                        peaks = detectors.two_average_detector(signal)

                        for i in range(len(peaks) - 1):
                            new = peaks[i + 1] - peaks[i]
                            RRi.append(new)

                        Samp_entropy.append(nk.entropy_sample(RRi))
                        fuzzy_entropy.append(nk.entropy_fuzzy(RRi))
                        shannon_entropy.append(nk.entropy_shannon(RRi))
                        approximate_entropy.append(nk.entropy_approximate(RRi))
                        HRV = np.append(HRV, RRi)
                        RRi = []
                        start = start + (10 * fs)
                        end = end + (10 * fs)
                        i += 1

                    # ==  ploting the entropy values in the same figure
                    ymin = 3.5
                    ymax = 8.5

                    print(approximate_entropy)
                    fig, axs = plt.subplots()
                    axs.plot(shannon_entropy, label="Shanon entropy", marker='o')
                    axs.set_xlabel('time in min')
                    axs.set_ylabel('Amplitude')
                    axs.grid(True)
                    axs.set_ylim([ymin, ymax])
                    axs.legend()

                    ymin = 0
                    ymax = 2

                    fig, axs = plt.subplots()
                    axs.plot(Samp_entropy, label="Sample entropy", marker='o')
                    axs.set_xlabel('time in min')
                    axs.set_ylabel('Amplitude')
                    axs.grid(True)
                    axs.set_ylim([ymin, ymax])
                    axs.legend()

                    ymin = 0
                    ymax = 1.5

                    fig, axs = plt.subplots()
                    axs.plot(fuzzy_entropy, label="Fuzzy entropy", marker='o')
                    axs.set_xlabel('time in min')
                    axs.set_ylabel('Amplitude')
                    axs.grid(True)
                    axs.set_ylim([ymin, ymax])
                    axs.legend()

                    fig, axs = plt.subplots()
                    axs.plot(approximate_entropy, label="Approximate entropy normally", marker='o')
                    axs.set_xlabel('time in min')
                    axs.set_ylabel('Amplitude')
                    axs.grid(True)
                    axs.set_ylim([ymin, ymax])
                    axs.legend()

                    fig, axs = plt.subplots()
                    axs.plot(signal_input, label="the input signal")
                    axs.set_ylabel('Amplitude')
                    axs.grid(True)
                    axs.legend()

                    detectors = Detectors(fs)
                    peaks = detectors.two_average_detector(signal_output)

                    pl = [0] * len(peaks)
                    fig, axs = plt.subplots()
                    axs.plot(signal_input, label="the input signal")
                    axs.plot(peaks, pl, 'ro')
                    axs.set_ylabel('Amplitude')
                    axs.grid(True)
                    axs.legend()

                    plt.show()

                except:
                    messagebox.showinfo("Attension",
                                        "Vous devez changer la valeur du filtre inotch en fonction du signal d'entrée 50Hz ou 60Hzs")

            else:
                messagebox.showinfo("Attension",
                                    "la partie sélectionnée dépassant la longueur du signal d'entrée")

        def Fermer():
            plt.close(1)
            plt.close(2)
            plt.close(3)
            plt.close(4)
            plt.close(5)
            plt.close(6)
            plt.close(7)

        # == interface initialization
        title = Label(self.root, text="Manipulation de signal ", bd=10, relief=GROOVE,font=("times new roman", 40, "bold"), bg="#DCDCDC", fg="#191970")
        title.pack(side=TOP, fill=X)
        btn_Frame = Frame(self.root, bd=4, relief=RIDGE)
        btn_Frame.place(x=50, y=100, width=1400)

        ################################
        # == signal selection part
        global path_browse
        btn_charge_image_seg = Button(btn_Frame, text="selectionner le signal", width=30, command=Browse)
        btn_charge_image_seg.grid(row=0, column=0,padx=20,pady=10)
        path_browse = tk.Label(btn_Frame, text="Le chemain de signal sélectionner")
        path_browse.grid(row=0, column=1,padx=20,pady=10)

        ################################
        # == electrode part
        global electrode
        path_seg = Label(btn_Frame, text="electrode à selectionner", bd=10, font=("times new roman", 20, "bold"),
                         fg="#191970").grid(
            row=1, column=0, padx=20, pady=10, sticky="w")
        electrode = Entry(btn_Frame, font=("times new roman", 15, "bold"), bd=5, relief=GROOVE)
        electrode.grid(row=1, column=1, pady=10, padx=20, sticky="w")

        ################################
        # == fréqunce d'échantillonnage
        path_recon = Label(btn_Frame, text="fréquence d'échantillonnage", bd=10, font=("times new roman", 18, "bold"),fg="#191970").grid(row=2, column=0, padx=20, pady=10, sticky="w")
        global Fss
        Fss = tk.Entry(btn_Frame, font=("times new roman", 15, "bold"), bd=5, relief=GROOVE)
        Fss.grid(row=2, column=1,pady=30, padx=20,sticky="w")

        ################################
        # == fréqunce d'échantillonnage
        path_recon = Label(btn_Frame, text="Valeur de filtre notch", bd=10, font=("times new roman", 18, "bold"),fg="#191970").grid(row=2, column=2, padx=20, pady=10, sticky="w")
        global notch

        notch = tk.Entry(btn_Frame, font=("times new roman", 15, "bold"), bd=5, relief=GROOVE)
        notch.grid(row=2, column=3, pady=30, padx=20, sticky="w")

        ################################
        # == ROI of the input signal

        # == start
        start_signal = Label(btn_Frame, text="début en minute", bd=10, font=("times new roman", 18, "bold"),fg="#191970").grid(row=3, column=0, padx=10, pady=10, sticky="w")
        start_entry = tk.Entry(btn_Frame, font=("times new roman", 15, "bold"), bd=5, relief=GROOVE, text='start')
        start_entry.grid(row=3, column=1, pady=10, padx=20,sticky="w")
        ###################
        # == end
        end_signal = Label(btn_Frame, text="Fin en minute", bd=10, font=("times new roman", 18, "bold"),fg="#191970").grid(row=3, column=2, padx=20, pady=10, sticky="w")
        end_entry = tk.Entry(btn_Frame, font=("times new roman", 15, "bold"), bd=5, relief=GROOVE)
        end_entry.grid(row=3, column=3, pady=10, padx=20,sticky="w")
        ################################
        # == extractino of the entropy values from the input ECG signal (Manef's work)
        global path_signal
        entropie_button = Button(btn_Frame, text="calcule de l'entropie",height=1 ,width=50,command=lambda: Entropie(path_signal))
        entropie_button.grid(row=4, column=1, padx=0, pady=10)

        ################################
        btn_cont = Button(btn_Frame, text=" ...", height=1 ,width=50).grid(row=5, column=1, padx=20, pady=10)
        btn_CC = Button(btn_Frame, text="...",height=1 ,width=50).grid(row=6, column=1, padx=10, pady=10)
        btn_V = Button(btn_Frame, text="...", height=1,width=50).grid(row=7, column=1, padx=10, pady=10)
        btn_s = Button(btn_Frame, text="...", height=1 ,width=50).grid(row=8, column=1, padx=10, pady=10)
        btn_ferme = Button(btn_Frame, text="Fermer", height=1 ,width=50,command=lambda: Fermer()).grid(row=9, column=1, padx=10, pady=10)
        ################################



root = Tk()
ob = ReconstructuionSegmentation(root)
root.mainloop()
