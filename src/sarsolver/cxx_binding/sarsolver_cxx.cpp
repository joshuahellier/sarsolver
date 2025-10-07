#include "sarsolver_cxx.hpp"
#include <cmath>
#include <complex>
#include <iostream>
#include "fftw3.h"

const std::complex<double> J = std::complex<double>(0.0, 1.0);
const std::complex<double> two_pi_J = std::complex<double>(0.0, 2.0 * M_PI);

double distance(const ThreeVector &x, const ThreeVector &y) {
    double dsquared = 0;
    for (int axis = 0; axis < 3; ++axis) {
        dsquared += (x.contents[axis] - y.contents[axis]) * (x.contents[axis] - y.contents[axis]);
    }
    return sqrt(dsquared);
}

double bistatic_range(ThreeVector &trans_pos, ThreeVector &recv_pos, ThreeVector &x) {
    return distance(trans_pos, x) + distance(recv_pos, x);
}

inline size_t math_modulo(long a, long b) {
    return ((a % b) + b) % b;
}

void forward_evaluate(SarCalculationInfo &sar_calc_info) {
    auto sar_worker = SarWorker(sar_calc_info);
    sar_worker.setup_forward_evaluate();
    sar_worker.execute_forward_evaluate();
}

void adjoint_evaluate(SarCalculationInfo &sar_calc_info) {
    auto sar_worker = SarWorker(sar_calc_info);
    sar_worker.setup_adjoint_evaluate();
    sar_worker.execute_adjoint_evaluate();
}

void roundabout_copy(SarCalculationInfo &in, SarCalculationInfo &out) {
    auto sar_worker = SarWorker(in);
    sar_worker.copy_into_struct(out);
}

void forward_copy(SarCalculationInfo &in, SarCalculationInfo &out) {
    auto sar_worker = SarWorker(in);
    sar_worker.setup_forward_evaluate();
    sar_worker.execute_forward_evaluate();
    sar_worker.copy_into_struct(out);
}

void adjoint_copy(SarCalculationInfo &in, SarCalculationInfo &out) {
    auto sar_worker = SarWorker(in);
    sar_worker.setup_adjoint_evaluate();
    sar_worker.execute_adjoint_evaluate();
    sar_worker.copy_into_struct(out);
}

void direct_copy(SarCalculationInfo &in, SarCalculationInfo &out) {
    out.num_fast_times = in.num_fast_times;
    out.num_slow_times = in.num_slow_times;
    out.num_scatterers = in.num_scatterers;
    out.centre_frequency = in.centre_frequency;
    out.sample_frequency = in.sample_frequency;
    out.c_eff = in.c_eff;
    out.upsample_ratio = in.upsample_ratio;
    out.sign_multiplier = in.sign_multiplier;
    std::cout << "Copying:" << std::endl;
    for (size_t s = 0; s < in.num_slow_times; ++s) {
        out.slow_time_weighting[s] = in.slow_time_weighting[s];
        out.slow_time_weighting[s + 1] = in.slow_time_weighting[s + 1];
        for (size_t i = 0; i < 3; ++i) {
            out.transmit_posns[3 * s + i] = in.transmit_posns[3 * s + i];
            out.receive_posns[3 * s + i] = in.transmit_posns[3 * s + i];
            out.stab_ref_posns[3 * s + i] = in.stab_ref_posns[3 * s + i];
        }
        for (size_t f = 0; f < in.num_fast_times; ++f) {
            out.phase_history[2 * (in.num_fast_times * s + f)] = in.phase_history[2 * (in.num_fast_times * s + f)];
            out.phase_history[2 * (in.num_fast_times * s + f) + 1] = in.phase_history[2 * (in.num_fast_times * s + f) + 1];
        }
    }
    std::cout << "Slow time worked." << std::endl;
    for (size_t f = 0; f < in.num_fast_times; ++f) {
        out.waveform_fft[2 * f] = in.waveform_fft[2 * f];
        out.waveform_fft[2 * f + 1] = in.waveform_fft[2 * f + 1];
    }
    std::cout << "Fast time worked." << std::endl;
    for (size_t n = 0; n < in.num_scatterers; ++n) {
        out.scattering_amplitudes[2 * n] = in.scattering_amplitudes[2 * n];
        out.scattering_amplitudes[2 * n + 1] = in.scattering_amplitudes[2 * n + 1];
        for (size_t i = 0; i < 3; ++i) {
            out.scat_posns[3 * n + i] = in.scat_posns[3 * n + i];
        }
    }
    std::cout << "Scatterers worked. All complete." << std::endl;
}


SarMeasurements::SarMeasurements(size_t fast_times, size_t slow_times, double centre_frequency, double sample_frequency, double c_eff) {
    num_fast_times = fast_times;
    num_slow_times = slow_times;
    centre_freq = centre_frequency;
    sample_freq = sample_frequency;
    light_speed = c_eff;
    own_memory = true;
    transmit_posns = new ThreeVector[num_slow_times];
    receive_posns = new ThreeVector[num_slow_times];
    stab_ref_posns = new ThreeVector[num_slow_times];
    phase_history = new std::complex<double>[num_slow_times * num_fast_times];
}

SarMeasurements::SarMeasurements(SarCalculationInfo &sar_calc_info) {
    num_fast_times = sar_calc_info.num_fast_times;
    num_slow_times = sar_calc_info.num_slow_times;
    centre_freq = sar_calc_info.centre_frequency;
    sample_freq = sar_calc_info.sample_frequency;
    light_speed = sar_calc_info.c_eff;
    own_memory = false;
    transmit_posns = reinterpret_cast<ThreeVector *>(sar_calc_info.transmit_posns);
    receive_posns = reinterpret_cast<ThreeVector *>(sar_calc_info.receive_posns);
    stab_ref_posns = reinterpret_cast<ThreeVector *>(sar_calc_info.stab_ref_posns);
    phase_history = reinterpret_cast<std::complex<double> *>(sar_calc_info.phase_history);
}

SarMeasurements::~SarMeasurements() {
    if (own_memory) {
        delete[] transmit_posns;
        delete[] receive_posns;
        delete[] stab_ref_posns;
        delete[] phase_history;
    }
}

void SarMeasurements::copy_into_struct(SarCalculationInfo &sar_calc_info) const {
    sar_calc_info.c_eff = light_speed;
    sar_calc_info.centre_frequency = centre_freq;
    sar_calc_info.sample_frequency = sample_freq;
    sar_calc_info.num_fast_times = num_fast_times;
    sar_calc_info.num_slow_times = num_slow_times;
    auto transmit_posns_r = sar_calc_info.transmit_posns;
    auto receive_posns_r = sar_calc_info.receive_posns;
    auto srps_r = sar_calc_info.stab_ref_posns;
    auto phase_history_r = sar_calc_info.phase_history;
    for (size_t s = 0; s < num_slow_times; ++s) {
        for (size_t i = 0; i < 3; i++) {
            *transmit_posns_r = transmit_posns[s].contents[i];
            ++transmit_posns_r;
            *receive_posns_r = receive_posns[s].contents[i];
            ++receive_posns_r;
            *srps_r = stab_ref_posns[s].contents[i];
            ++srps_r;
        }
        for (size_t f = 0; f < num_fast_times; ++f) {
            *phase_history_r = std::real(phase_history[num_fast_times * s + f]);
            ++phase_history_r;
            *phase_history_r = std::imag(phase_history[num_fast_times * s + f]);
            ++phase_history_r;
        }
    }
}

SarBornHypothesis::SarBornHypothesis(size_t scatterers) {
    num_scatterers = scatterers;
    own_memory = true;
    scat_posns = new ThreeVector[num_scatterers];
    scat_amps = new std::complex<double>[num_scatterers];
}

SarBornHypothesis::SarBornHypothesis(SarCalculationInfo &sar_calc_info) {
    num_scatterers = sar_calc_info.num_scatterers;
    own_memory = false;
    scat_posns = reinterpret_cast<ThreeVector *>(sar_calc_info.scat_posns);
    scat_amps = reinterpret_cast<std::complex<double> *>(sar_calc_info.scattering_amplitudes);
}

SarBornHypothesis::~SarBornHypothesis() {
    if (own_memory) {
        delete[] scat_amps;
        delete[] scat_posns;
    }
}

void SarBornHypothesis::copy_into_struct(SarCalculationInfo &sar_calc_info) const {
    sar_calc_info.num_scatterers = num_scatterers;
    for (size_t s = 0; s < num_scatterers; ++s) {
        for (size_t i = 0; i < 3; i++) {
            sar_calc_info.scat_posns[3 * s + i] = scat_posns[s].contents[i];
        }
        sar_calc_info.scattering_amplitudes[2 * s] = std::real(scat_amps[s]);
        sar_calc_info.scattering_amplitudes[2 * s + 1] = std::imag(scat_amps[s]);
    }
}


SarWorker::SarWorker(size_t fast_times, size_t working_fast_times, size_t slow_times, size_t scatterers, size_t index,
                     double centre_frequency, double sample_frequency, double sign, double c_eff) {
    measurements = SarMeasurements(fast_times, slow_times, centre_frequency, sample_frequency, c_eff);
    hypotheses = SarBornHypothesis(scatterers);
    working_num_fast_times = working_fast_times;
    worker_index = index;
    centre_wavenumber = 2.0 * M_PI * centre_frequency / c_eff;
    working_spatial_sample_rate = (sample_frequency * (double) (working_num_fast_times)) / ((double) (measurements.num_fast_times) * c_eff);
    sign_multiplier = sign;

    range_profile_fft = new std::complex<double>[measurements.num_fast_times];
    waveform_fft = new std::complex<double>[measurements.num_fast_times];
    working_k_modes = new std::complex<double>[working_num_fast_times];
    working_range_profile = new std::complex<double>[working_num_fast_times];
    slow_time_weighting = new std::complex<double>[measurements.num_slow_times];

    forward_fft_plan = fftw_plan_dft_1d(working_num_fast_times,
                                        reinterpret_cast<fftw_complex *>(working_range_profile),
                                        reinterpret_cast<fftw_complex *>(working_k_modes), FFTW_FORWARD,
                                        FFTW_MEASURE);
    inverse_fft_plan = fftw_plan_dft_1d(working_num_fast_times,
                                        reinterpret_cast<fftw_complex *>(working_k_modes),
                                        reinterpret_cast<fftw_complex *>(working_range_profile), FFTW_BACKWARD,
                                        FFTW_MEASURE);
}

SarWorker::SarWorker(SarCalculationInfo &sar_calc_info) {
    measurements = SarMeasurements(sar_calc_info);
    hypotheses = SarBornHypothesis(sar_calc_info);
    worker_index = 0;
    working_num_fast_times = sar_calc_info.upsample_ratio * sar_calc_info.num_fast_times;

    working_spatial_sample_rate = (sar_calc_info.sample_frequency *
                                   (double) (working_num_fast_times)) / ((double) (measurements.num_fast_times) * sar_calc_info.c_eff);
    sign_multiplier = sar_calc_info.sign_multiplier;
    centre_wavenumber = 2.0 * sign_multiplier * M_PI * sar_calc_info.centre_frequency / sar_calc_info.c_eff;

    range_profile_fft = new std::complex<double>[measurements.num_fast_times];
    working_k_modes = new std::complex<double>[working_num_fast_times];
    working_range_profile = new std::complex<double>[working_num_fast_times];
    waveform_fft = new std::complex<double>[measurements.num_fast_times];
    slow_time_weighting = new std::complex<double>[measurements.num_slow_times];
    auto helpful_pointer = reinterpret_cast<std::complex<double> *>(sar_calc_info.waveform_fft);
    for (size_t f = 0; f < measurements.num_fast_times; ++f) {
        waveform_fft[f] = helpful_pointer[f];
    }
    auto helpful_pointer2 = reinterpret_cast<std::complex<double> *>(sar_calc_info.slow_time_weighting);
    for (size_t s = 0; s < measurements.num_slow_times; ++s) {
        slow_time_weighting[s] = helpful_pointer2[s];
    }

    forward_fft_plan = fftw_plan_dft_1d(working_num_fast_times,
                                        reinterpret_cast<fftw_complex *>(working_range_profile),
                                        reinterpret_cast<fftw_complex *>(working_k_modes), FFTW_FORWARD,
                                        FFTW_ESTIMATE);
    inverse_fft_plan = fftw_plan_dft_1d(working_num_fast_times,
                                        reinterpret_cast<fftw_complex *>(working_k_modes),
                                        reinterpret_cast<fftw_complex *>(working_range_profile), FFTW_BACKWARD,
                                        FFTW_ESTIMATE);
}

SarWorker::~SarWorker() {
    delete[] waveform_fft;
    delete[] working_k_modes;
    delete[] working_range_profile;
    delete[] range_profile_fft;
    delete[] slow_time_weighting;
    fftw_destroy_plan(forward_fft_plan);
    fftw_destroy_plan(inverse_fft_plan);
}

void SarWorker::setup_forward_evaluate() const {
    for (size_t i = 0; i < measurements.num_fast_times * measurements.num_slow_times; ++i) {
        measurements.phase_history[i] = std::complex<double>(0.0, 0.0);
    }
    zero_fft_buffers();
}

void SarWorker::execute_forward_evaluate() {
    double by_dx = working_spatial_sample_rate;
    size_t half_num_fast_times = measurements.num_fast_times / 2;
    long fast_time_index;
    std::complex<double> amplitude;
    ThreeVector scat_posn, transmit_pos, receive_pos, stab_ref_pos;
    std::complex<double> scat_amp;
    double differential_bistatic_range, srp_bistatic_range;
    for (size_t slow_time_index = 0; slow_time_index < measurements.num_slow_times; ++slow_time_index) {
        transmit_pos = measurements.transmit_posns[slow_time_index];
        receive_pos = measurements.receive_posns[slow_time_index];
        stab_ref_pos = measurements.stab_ref_posns[slow_time_index];
        srp_bistatic_range = bistatic_range(transmit_pos, receive_pos, stab_ref_pos);
        for (size_t i = 0; i < hypotheses.num_scatterers; ++i) {
            scat_posn = hypotheses.scat_posns[i];
            scat_amp = hypotheses.scat_amps[i];
            differential_bistatic_range = bistatic_range(transmit_pos, receive_pos, scat_posn) - srp_bistatic_range;
            amplitude = exp(J * differential_bistatic_range * centre_wavenumber) * scat_amp * slow_time_weighting[slow_time_index];
            fast_time_index = math_modulo((long) std::round(differential_bistatic_range * by_dx), (long) (working_num_fast_times));
            working_range_profile[fast_time_index] += amplitude;
        }
        fftw_execute(forward_fft_plan);
        // Here we copy the relevant FFTd data into the downsampled measurement output data, multiplying by the waveform
        // spectral support as we go. If there's an odd Fourier mode left over in the middle, it just remains 0.
        for (size_t i = 0; i < half_num_fast_times; ++i) {
            // Going backwards to halfway:
            measurements.phase_history[measurements.num_fast_times * slow_time_index +
                                       measurements.num_fast_times - i - 1] =
                    waveform_fft[measurements.num_fast_times - i - 1] * working_k_modes[working_num_fast_times - i - 1];
            // Going forwards to halfway:
            measurements.phase_history[measurements.num_fast_times * slow_time_index + i] = waveform_fft[i]
                                                                                            * working_k_modes[i];
        }
        if(measurements.num_fast_times % 2 == 1){
        measurements.phase_history[measurements.num_fast_times * slow_time_index + half_num_fast_times] =
                waveform_fft[half_num_fast_times] *
                working_k_modes[half_num_fast_times];}
        zero_fft_buffers();
    }
}

void SarWorker::setup_adjoint_evaluate() const {
    for (size_t i = 0; i < hypotheses.num_scatterers; ++i) {
        hypotheses.scat_amps[i] = std::complex<double>(0.0, 0.0);
    }
    zero_fft_buffers();
}

void SarWorker::execute_adjoint_evaluate() {
    double by_dx = working_spatial_sample_rate;
    double differential_bistatic_range, srp_bistatic_range;
    size_t half_num_fast_times = measurements.num_fast_times / 2;
    long fast_time_index;
    std::complex<double> amplitude;
    ThreeVector transmit_pos, receive_pos, stab_ref_pos;
    for (size_t slow_time_index = 0; slow_time_index < measurements.num_slow_times; ++slow_time_index) {
        transmit_pos = measurements.transmit_posns[slow_time_index];
        receive_pos = measurements.receive_posns[slow_time_index];
        stab_ref_pos = measurements.stab_ref_posns[slow_time_index];
        srp_bistatic_range = bistatic_range(transmit_pos, receive_pos, stab_ref_pos);
        for (size_t i = 0; i < half_num_fast_times; ++i) {
            // Going forwards to halfway:
            working_k_modes[i] = std::conj(waveform_fft[i]) * measurements.phase_history[measurements.num_fast_times * slow_time_index + i];
            // Going backwards to halfway:
            working_k_modes[working_num_fast_times - i - 1] = std::conj(waveform_fft[measurements.num_fast_times - i - 1])
                                                              * measurements.phase_history[measurements.num_fast_times * slow_time_index +
                                                                                           measurements.num_fast_times - i - 1];
        }
        if(measurements.num_fast_times % 2 == 1){
            working_k_modes[half_num_fast_times] = std::conj(waveform_fft[half_num_fast_times]) *
                                                   measurements.phase_history[measurements.num_fast_times * slow_time_index + half_num_fast_times];
        }

        fftw_execute(inverse_fft_plan);
        for (std::size_t scat_index = 0; scat_index < hypotheses.num_scatterers; ++scat_index) {
            differential_bistatic_range = bistatic_range(transmit_pos, receive_pos, hypotheses.scat_posns[scat_index]) - srp_bistatic_range;
            fast_time_index = math_modulo((long) std::round(0.5 + differential_bistatic_range * by_dx), (long) (working_num_fast_times));
            amplitude = working_range_profile[fast_time_index] * exp(-J * differential_bistatic_range * centre_wavenumber);
            hypotheses.scat_amps[scat_index] += amplitude * slow_time_weighting[slow_time_index];
        }
        zero_fft_buffers();
    }
}

void SarWorker::copy_into_struct(SarCalculationInfo &sar_calc_info) const {
    measurements.copy_into_struct(sar_calc_info);
    hypotheses.copy_into_struct(sar_calc_info);
    for (size_t f = 0; f < measurements.num_fast_times; ++f) {
        sar_calc_info.waveform_fft[2 * f] = std::real(waveform_fft[f]);
        sar_calc_info.waveform_fft[2 * f + 1] = std::imag(waveform_fft[f]);
    }
    for (size_t s = 0; s < measurements.num_slow_times; ++s) {
        sar_calc_info.slow_time_weighting[2 * s] = std::real(slow_time_weighting[s]);
        sar_calc_info.slow_time_weighting[2 * s + 1] = std::imag(slow_time_weighting[s]);
    }
}

void SarWorker::zero_fft_buffers() const {
    for (size_t i = 0; i < working_num_fast_times; ++i) {
        working_range_profile[i] = std::complex<double>(0.0, 0.0);
        working_k_modes[i] = std::complex<double>(0.0, 0.0);
    }
}
