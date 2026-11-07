/* Scalar per-step maths of the HVAC devices: air handler, VAV, boiler, hot
 * water system, pump.
 *
 * Every function here is a transcription of one method in
 * sims/<tree>/smart_buildings/smart_control/simulator/, with the physical
 * constants passed in rather than hardcoded so the same library serves both
 * simulator trees and cannot drift from their constants module.
 *
 * Expression order follows the Python exactly (left-to-right, no
 * reassociation, no FMA): with -ffp-contract=off the results are bit-identical
 * for every input the simulator produces.
 *
 * Nothing here holds state. The two stateful pieces of the boiler -- the tank
 * temperature follower and the corrected tree's energy balance -- are exposed
 * as pure functions over the state the caller owns.
 */

#ifndef CENV_HVAC_H
#define CENV_HVAC_H

#ifdef __cplusplus
extern "C" {
#endif

/* -- air_handler.AirHandler ------------------------------------------------ */

/* get_mixed_air_temp: recirculation r of return air at `recirc_temp`, the rest
 * outside air at `ambient`. */
double cenv_hvac_mixed_air_temp(double recirculation, double recirc_temp,
                                double ambient);

/* get_supply_air_temp: the coil drives the mixed air down to the supply
 * setpoint when it is warmer than that and the unit is running (an air handler
 * only ever cools; heating is the reheat coils). */
double cenv_hvac_supply_air_temp(double recirculation, double recirc_temp,
                                 double ambient, double supply_setpoint,
                                 int running);

/* compute_thermal_energy_rate: flow * rho * cp * (T_supply - T_mixed), W. */
double cenv_hvac_ahu_thermal_rate(double flow, double recirculation,
                                  double recirc_temp, double ambient,
                                  double supply_setpoint, double air_density,
                                  double air_cp, int running);

/* compute_fan_power: flow * dp / efficiency, W. */
double cenv_hvac_fan_power(double flow, double static_pressure,
                           double efficiency);

/* -- vav.Vav --------------------------------------------------------------- */

/* _compute_flow_rate_demand: damper * max_flow * sqrt(dp / dp_design), floored
 * at 1e-5 m3/s so the downstream divisions never see zero. */
double cenv_hvac_vav_flow_demand(double damper, double max_flow,
                                 double static_pressure,
                                 double design_static_pressure);

/* _reheat_effectiveness: min(1, C_water/C_air), capped so the coil cannot
 * discharge above `max_discharge_temp`. `have_temps` mirrors the Python's
 * optional supply/water temperatures. */
double cenv_hvac_reheat_effectiveness(double air_capacity_rate,
                                      double water_capacity_rate,
                                      double supply_air_temp,
                                      double water_temp,
                                      double max_discharge_temp,
                                      int have_temps);

/* compute_reheat_energy_rate: cp * rho * flow * (T_water - T_air), scaled by
 * the effectiveness, in the simulator's own left-to-right order. */
double cenv_hvac_reheat_energy_rate(double air_capacity_rate,
                                    double water_capacity_rate,
                                    double air_cp, double air_density,
                                    double air_flow,
                                    double supply_air_temp,
                                    double water_temp,
                                    double max_discharge_temp);

/* compute_zone_supply_temp: T_air + effectiveness * (T_water - T_air). */
double cenv_hvac_zone_supply_temp(double air_capacity_rate,
                                  double water_capacity_rate,
                                  double supply_air_temp,
                                  double water_temp,
                                  double max_discharge_temp);

/* compute_energy_applied_to_zone: the supply-air stream at the zone plus the
 * coil's duty, W. */
double cenv_hvac_zone_energy(double zone_temp, double air_flow,
                             double air_density, double air_cp,
                             double air_capacity_rate,
                             double water_capacity_rate,
                             double supply_air_temp, double water_temp,
                             double max_discharge_temp);

/* -- boiler.Boiler --------------------------------------------------------- */

/* compute_thermal_dissipation_rate: steady conduction through the tank wall in
 * series with convection off its outside, W. */
double cenv_hvac_tank_dissipation(double water_temp, double outside_temp,
                                  double tank_length, double tank_radius,
                                  double insulation_thickness,
                                  double insulation_conductivity,
                                  double convection_coefficient);

/* _adjust_temperature: the tank setpoint follower, K after `duration_s`. */
double cenv_hvac_tank_temperature(double setpoint, double actual,
                                  double duration_s, double heating_rate,
                                  double cooling_rate);

/* compute_thermal_energy_rate for a boiler without the energy balance: the
 * flow load, the standing loss and the tank's own charge/discharge, floored at
 * zero because a gas boiler cannot burn negative gas. */
double cenv_hvac_boiler_thermal_rate(double supply_setpoint, double return_temp,
                                     double outside_temp, double flow,
                                     double water_density, double water_cp,
                                     double water_capacity,
                                     double tank_temp_change,
                                     double duration_s,
                                     double tank_length, double tank_radius,
                                     double insulation_thickness,
                                     double insulation_conductivity,
                                     double convection_coefficient);

/* apply_heat_draw (corrected tree, boiler.py): one step of
 *   C dT/dt = Q_burner - Q_coils - Q_loss
 * Returns the burner heat rate in W; *tank_temperature is advanced in place. */
double cenv_hvac_tank_energy_balance(double *tank_temperature,
                                     double coil_heat_w, double outside_temp,
                                     double duration_s,
                                     double supply_setpoint,
                                     double water_capacity, double water_density,
                                     double water_cp, double heating_rate,
                                     double tank_length, double tank_radius,
                                     double insulation_thickness,
                                     double insulation_conductivity,
                                     double convection_coefficient);

/* -- hot_water_system.HotWaterSystem / pump.WaterPump --------------------- */

/* _calculate_flow_rate: Kv * sqrt(dp / (1 + Kv^2 * header_resistance)) / 3600,
 * m3/s. */
double cenv_hvac_water_flow_rate(double differential_pressure,
                                 double flow_factor, double header_resistance);

/* compute_pump_power: flow * rho * g * head / efficiency, W. */
double cenv_hvac_pump_power(double flow, double head,
                            double water_density, double gravity,
                            double efficiency);

/* head <-> pressure conversions used by the pump's setpoint properties. */
double cenv_hvac_head_to_pressure(double head, double water_density,
                                  double gravity, double pascals_per_bar);

double cenv_hvac_pressure_to_head(double pressure, double water_density,
                                  double gravity, double pascals_per_bar);

#ifdef __cplusplus
}
#endif

#endif /* CENV_HVAC_H */
