#ifndef SARSOLVER_CXX_HPP
#define SARSOLVER_CXX_HPP

#include <complex>
#include "fftw3.h"
#include <iostream>

const double c_0 = 299792458.0;

// Triplet of doubles, to store a 3-vector. Only need to do a few specific vector operations,
// so didn't want to import an external library for it.
struct ThreeVector {
    double contents[3];
};

// Calculates Euclidean distance between two 3-vectors.
double distance(const ThreeVector &x, const ThreeVector &y);

// Calculates bistatic range ||x - trans_pos|| + ||x - rec_pos|| .
double bistatic_range(ThreeVector &trans_pos, ThreeVector &recv_pos, ThreeVector &x);

// Computes a modulo b, and gives the right result for negative numbers (or at least, the one we want).
inline std::size_t math_modulo(long a, long b);

// Struct which stores all the ingredients necessary to perform simple scalar bistatic SAR forward/adjoint evaluations.
// Is bound on the Python-side.
extern "C" {
struct SarCalculationInfo {
    std::size_t num_fast_times, num_slow_times, num_scatterers;
    double *transmit_posns, *receive_posns, *stab_ref_posns, *scat_posns;
    double *phase_history, *scattering_amplitudes, *waveform_fft, *slow_time_weighting;
    double centre_frequency, sample_frequency, c_eff, upsample_ratio, sign_multiplier;
};
}

// Class for representing SAR measurements. Can be constructed from the relevant parts of a SarCalculationInfo
//struct and have its contents written out to one.
class SarMeasurements {
public:
    std::size_t num_fast_times, num_slow_times;
    ThreeVector *transmit_posns, *receive_posns, *stab_ref_posns; // num_slow_times x 3
    std::complex<double> *phase_history; // num_slow_times x num_fast_times
    bool own_memory;
    double centre_freq, sample_freq, light_speed;

    SarMeasurements(std::size_t fast_times, std::size_t slow_times, double centre_frequency, double sample_frequency,
                    double c_eff = c_0);

    explicit SarMeasurements(SarCalculationInfo &sar_calc_info);

    SarMeasurements() = default;

    ~SarMeasurements();

    void copy_into_struct(SarCalculationInfo &sar_calc_info) const;
};

// Class for representing SAR hypotheses (images). Can be constructed from the relevant parts of a SarCalculationInfo
//struct and have its contents written out to one.
class SarBornHypothesis {
public:
    std::size_t num_scatterers;
    ThreeVector *scat_posns; // num_scatterers x 3
    std::complex<double> *scat_amps; // num_scatterers long
    bool own_memory;

    explicit SarBornHypothesis(std::size_t scatterers);

    explicit SarBornHypothesis(SarCalculationInfo &sar_calc_info);

    SarBornHypothesis() = default;

    ~SarBornHypothesis();

    void copy_into_struct(SarCalculationInfo &sar_calc_info) const;
};

// Class that actually carries out forward and adjoint SAR evaluations. Constructed either by explicit feed of
// variables or from a SarCalculationInfo.
class SarWorker {
public:
    std::size_t worker_index, working_num_fast_times;
    SarMeasurements measurements;
    SarBornHypothesis hypotheses;
    double centre_wavenumber, working_spatial_sample_rate, sign_multiplier;
    std::complex<double> *range_profile_fft, *waveform_fft, *slow_time_weighting;
    std::complex<double> *working_k_modes, *working_range_profile;

    SarWorker(std::size_t fast_times, std::size_t working_fast_times, std::size_t slow_times, std::size_t scatterers,
              std::size_t index, double centre_frequency, double sample_frequency, double sign, double c_eff = c_0);

    explicit SarWorker(SarCalculationInfo &sar_calc_info);

    ~SarWorker();

    void execute_forward_evaluate();

    void execute_adjoint_evaluate();

    void setup_forward_evaluate() const;

    void setup_adjoint_evaluate() const;

    void copy_into_struct(SarCalculationInfo &sar_calc_info) const;

    void zero_fft_buffers() const;

private:
    fftw_plan forward_fft_plan, inverse_fft_plan;
};

extern "C" {
void forward_evaluate(SarCalculationInfo &sar_calc_info);
void adjoint_evaluate(SarCalculationInfo &sar_calc_info);
void roundabout_copy(SarCalculationInfo &in, SarCalculationInfo &out);
void forward_copy(SarCalculationInfo &in, SarCalculationInfo &out);
void adjoint_copy(SarCalculationInfo &in, SarCalculationInfo &out);
void direct_copy(SarCalculationInfo &in, SarCalculationInfo &out);
}

#endif //SARSOLVER_CXX_HPP
