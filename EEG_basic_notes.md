# What format is EEG data in?
An EDF Data file is a data format tailored for physiological signals. It is separated into a header and a data record.
The format is made so that it has 1 file per recording for each patient.
<img width="900" alt="Screenshot 2025-07-07 at 9 43 31 PM" src="https://github.com/user-attachments/assets/5241d0f1-f21a-4c08-a839-de14e0d16f7e" />
Basically, the header has patient info while the data record has data.
# PSD
Average band power is a single number that describes how much a particular frequency range contributes to the overall power of the signal. Useful when we want to have a single number to summarize a key feature of my data.
A periodogram basically breaks down a signal into frequency components and show the power or strength of each frequency range.
PSD (Power spectral density): the concentration of a signal's power per unit of frequency. Shows which frequencies contain the most energy

# EEG NOTES START HERE

Hans Berger considered pioneer. He believed that brain wavesgave rise to telepathy.
Alpha (~10 Hz) and Beta (faster, lower range) waves.

## How are these brain waves generated?

3-256 electrodes on the scalp. Electrodes are tied to an amplifier that enhances the signal before it is transformed into digital data in the computer.

Non-invasive EEG Data- represents the activity of collective neurons. Simplified version of brain activity (the symphony).

Activity that an electrode measures may not be the area directly under it, but currents from neurons from other areas as well.
** Important **
Electric Dipoles. Dipoles are something with two ends or poles that are opposing each other. In electricity. A dopole is something that holds a positive charge and negative charge at two ends.

Brain areas generating signals present dipole configuration. This happens when a large group of neurons have synchronized activity. Forming an equivalent current dipole (ECD) and creating a dipolar field with one end positive and one end negative charge. Brain's cortex contains many pyramidal neurons that creat measurable ECD's. 

Group of neurons physically align and synchronizing can cause eeg signals called open fields. Other configurations inhibiting each other's activity are called closed fields - do not generate detectable signals on the scalp.

2 types of neural activities:
Action potential and post-synaptic potential.
AP - when membrane potential > threshold potential -> axonal spike. Brevity in time, wide synchronization, slower pac e
PSP -  electric potentials on some membranes influenced by other neurons. Including excitatory and inhibitory signals that converge onto it.
These inputs sum over time, allowing increased synchronization of PSPs across multiple neurons and creating a collective signal detectable on scalp. When an electrode faces an open field, where (sync), results in a stronger EEG signal generated.

Dipole forms between cell body and axon because during an AP, there are localized regions of high concentrations of positive ions, and other regions are negative relative to it.

In the sulculs (groove), there is less synchronized activity but in gyrus, there is more sync'ed activity -> stronger signal.

# How to record signals using EEG

Electrodes - Simple sensors made of coductive material connected to wires and connected to amplifiers. E.g. tin, silver, silver chloride, gold, graphite. In clinical settings, electrodes often placed on scalp using sticky paste. In research settings, caps are used with electrodes fixed in positions. Electrodes don't touch the scalp. It maintains a few mm of distance from it. Conductive gel is actually injected in between to maintain conductivity. Dry electrodes (directly placed on scalp) are more prone to noise. Wet electrodes are better data quality.

"The International 10-20 system is the standard method for electrode placement in electroencephalography (EEG). This system utilizes anatomical landmarks on the skull to standardize electrode positions, ensuring reproducibility of EEG recordings across different individuals and laboratories." - Gemini. This convention helps ensure consistency across studies.
<img width="300" alt="Screenshot 2025-07-08 at 10 48 36 PM" src="https://github.com/user-attachments/assets/281c6bd2-5842-4117-acee-28788f9ff55d" />
Nasion, Inion, vertex.
Electrode positions tend to be more approximate, because of different head shapes. The letters on the image represent the different regimes on the head.
e.g. F - frontal. P- parietal, T- temporal, O-occipital.
C-central. Odd numbers = left hemisphere. F3- left frontal lobe. T6, right temporal lobe.

z designation represents 0. Cz is the center point for reference.
## How does EEG capture and process?
Amplifiers amplify the little signals collected by the electrodes from the scalp. Amplifiers convert continuous signals to discrete signals (voltages at specific time). 

# Electrical potential
- Potential for the movement of current (flow of charge)
- Electric potential = the voltage difference in positive and negative terminals.
- We have a cup of water. The amount it can spill when tipped over is called potential. But when it actually spills, the amount flowing over time is called the current. Potential measures the difference between the water in container and the water on the ground.
- Electric potential is the difference of the object compared to GND (ground). Assumed that ground has zero electric charge.
- In EEG recording, we are measuring the electric potential diff. between the elctrodes on the scalp (mV), meaning the diff. between that electrode and ground. A ground electrode is used as the reference point. The setup unavoidable includes noise from the static electricity and stuff.
EEG uses differential amplifiers:
Active elctrode, ground electrode, and a reference electrode.
(A-G)-(R-G) I think this eliminates noise from ground electrode and environment.

# What's a good EEG signal?







