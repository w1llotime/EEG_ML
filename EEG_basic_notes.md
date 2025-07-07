An EDF Data file is a data format tailored for physiological signals. It is separated into a header and a data record.
The format is made so that it has 1 file per recording for each patient.
<img width="1405" alt="Screenshot 2025-07-07 at 9 43 31 PM" src="https://github.com/user-attachments/assets/5241d0f1-f21a-4c08-a839-de14e0d16f7e" />
Basically, the header has patient info while the data record has data.
# PSD
Average band power is a single number that describes how much a particular frequency range contributes to the overall power of the signal. Useful when we want to have a single number to summarize a key feature of my data.
A periodogram basically breaks down a signal into frequency components and show the power or strength of each frequency range.
PSD (Power spectral density): the concentration of a signal's power per unit of frequency. Shows which frequencies contain the most energy
