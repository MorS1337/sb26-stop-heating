/* HVAC device maths. See hvac.h for the provenance of every function. */

#include <math.h>
#include <stddef.h>

#include "hvac.h"

double cenv_hvac_mixed_air_temp(double recirculation, double recirc_temp,
                                double ambient) {
  return recirculation * recirc_temp + (1.0 - recirculation) * ambient;
}

double cenv_hvac_supply_air_temp(double recirculation, double recirc_temp,
                                 double ambient, double supply_setpoint,
                                 int running) {
  const double mixed = cenv_hvac_mixed_air_temp(recirculation, recirc_temp,
                                                ambient);
  if (mixed > supply_setpoint && running) {
    return supply_setpoint;
  }
  return mixed;
}

double cenv_hvac_ahu_thermal_rate(double flow, double recirculation,
                                  double recirc_temp, double ambient,
                                  double supply_setpoint, double air_density,
                                  double air_cp, int running) {
  const double mixed = cenv_hvac_mixed_air_temp(recirculation, recirc_temp,
                                                ambient);
  const double supply = cenv_hvac_supply_air_temp(recirculation, recirc_temp,
                                                 ambient, supply_setpoint,
                                                 running);
  return flow * air_density * air_cp * (supply - mixed);
}

double cenv_hvac_fan_power(double flow, double static_pressure,
                           double efficiency) {
  return flow * static_pressure / efficiency;
}

double cenv_hvac_vav_flow_demand(double damper, double max_flow,
                                 double static_pressure,
                                 double design_static_pressure) {
  const double demand =
      damper * max_flow * sqrt(static_pressure / design_static_pressure);
  return (demand > 0.00001) ? demand : 0.00001;
}

double cenv_hvac_reheat_effectiveness(double air_capacity_rate,
                                      double water_capacity_rate,
                                      double supply_air_temp,
                                      double water_temp,
                                      double max_discharge_temp,
                                      int have_temps) {
  if (air_capacity_rate <= 0.0) {
    return 0.0;
  }
  double effectiveness = water_capacity_rate / air_capacity_rate;
  if (effectiveness > 1.0) {
    effectiveness = 1.0;
  }
  if (!have_temps) {
    return effectiveness;
  }
  const double gap = water_temp - supply_air_temp;
  if (gap <= 0.0) {
    return effectiveness;
  }
  const double headroom = max_discharge_temp - supply_air_temp;
  if (headroom <= 0.0) {
    return 0.0;
  }
  const double cap = headroom / gap;
  return (cap < effectiveness) ? cap : effectiveness;
}

double cenv_hvac_reheat_energy_rate(double air_capacity_rate,
                                    double water_capacity_rate,
                                    double air_cp, double air_density,
                                    double air_flow,
                                    double supply_air_temp,
                                    double water_temp,
                                    double max_discharge_temp) {
  const double effectiveness = cenv_hvac_reheat_effectiveness(
      air_capacity_rate, water_capacity_rate, supply_air_temp, water_temp,
      max_discharge_temp, 1);
  /* Same left-to-right order as vav.compute_reheat_energy_rate:
   * effectiveness * cp * rho * flow * (T_water - T_air). */
  return effectiveness * air_cp * air_density * air_flow *
         (water_temp - supply_air_temp);
}

double cenv_hvac_zone_supply_temp(double air_capacity_rate,
                                  double water_capacity_rate,
                                  double supply_air_temp,
                                  double water_temp,
                                  double max_discharge_temp) {
  const double effectiveness = cenv_hvac_reheat_effectiveness(
      air_capacity_rate, water_capacity_rate, supply_air_temp, water_temp,
      max_discharge_temp, 1);
  return supply_air_temp +
         effectiveness * (water_temp - supply_air_temp);
}

double cenv_hvac_zone_energy(double zone_temp, double air_flow,
                             double air_density, double air_cp,
                             double air_capacity_rate,
                             double water_capacity_rate,
                             double supply_air_temp, double water_temp,
                             double max_discharge_temp) {
  const double air_energy_rate =
      air_flow * air_density * air_cp * (supply_air_temp - zone_temp);
  const double reheat_energy_rate = cenv_hvac_reheat_energy_rate(
      air_capacity_rate, water_capacity_rate, air_cp, air_density, air_flow,
      supply_air_temp, water_temp, max_discharge_temp);
  return air_energy_rate + reheat_energy_rate;
}

double cenv_hvac_tank_dissipation(double water_temp, double outside_temp,
                                  double tank_length, double tank_radius,
                                  double insulation_thickness,
                                  double insulation_conductivity,
                                  double convection_coefficient) {
  if (water_temp < outside_temp) {
    return 0.0;
  }
  const double delta_temp = water_temp - outside_temp;
  const double numerator = tank_length * 2.0 * M_PI * delta_temp;
  const double interior_radius = tank_radius;
  const double exterior_radius = interior_radius + insulation_thickness;
  const double conduction_factor =
      log(exterior_radius / interior_radius) / insulation_conductivity;
  const double convection_factor =
      1.0 / convection_coefficient / exterior_radius;
  return numerator / (conduction_factor + convection_factor);
}

double cenv_hvac_tank_temperature(double setpoint, double actual,
                                  double duration_s, double heating_rate,
                                  double cooling_rate) {
  if (setpoint > actual) {
    const double heated = actual + heating_rate * duration_s / 60.0;
    return (heated < setpoint) ? heated : setpoint;
  }
  if (setpoint < actual) {
    const double cooled = actual - cooling_rate * duration_s / 60.0;
    return (cooled > setpoint) ? cooled : setpoint;
  }
  return setpoint;
}

double cenv_hvac_boiler_thermal_rate(double supply_setpoint, double return_temp,
                                     double outside_temp, double flow,
                                     double water_density, double water_cp,
                                     double water_capacity,
                                     double tank_temp_change,
                                     double duration_s,
                                     double tank_length, double tank_radius,
                                     double insulation_thickness,
                                     double insulation_conductivity,
                                     double convection_coefficient) {
  const double supply_water_temp =
      (supply_setpoint > return_temp) ? supply_setpoint : return_temp;
  const double flow_heating_energy_rate =
      water_cp * water_density * flow * (supply_water_temp - return_temp);
  const double dissipation_energy_rate = cenv_hvac_tank_dissipation(
      supply_water_temp, outside_temp, tank_length, tank_radius,
      insulation_thickness, insulation_conductivity, convection_coefficient);
  double tank_heating_energy_rate = 0.0;
  if (duration_s > 0.0) {
    tank_heating_energy_rate =
        water_cp * water_density * water_capacity * tank_temp_change /
        duration_s;
  }
  const double total = flow_heating_energy_rate + dissipation_energy_rate +
                       tank_heating_energy_rate;
  return (total > 0.0) ? total : 0.0;
}

double cenv_hvac_tank_energy_balance(double *tank_temperature,
                                     double coil_heat_w, double outside_temp,
                                     double duration_s,
                                     double supply_setpoint,
                                     double water_capacity, double water_density,
                                     double water_cp, double heating_rate,
                                     double tank_length, double tank_radius,
                                     double insulation_thickness,
                                     double insulation_conductivity,
                                     double convection_coefficient) {
  if (tank_temperature == NULL || duration_s <= 0.0) {
    return 0.0;
  }
  const double temperature = *tank_temperature;
  const double capacity = water_cp * water_density * water_capacity;
  const double loss = cenv_hvac_tank_dissipation(
      temperature, outside_temp, tank_length, tank_radius,
      insulation_thickness, insulation_conductivity, convection_coefficient);
  const double needed = capacity * (supply_setpoint - temperature) / duration_s
                        + coil_heat_w + loss;
  const double positive_coil = (coil_heat_w > 0.0) ? coil_heat_w : 0.0;
  const double max_rate =
      positive_coil + loss + capacity * heating_rate / 60.0;
  double burner = (needed > 0.0) ? needed : 0.0;
  if (burner > max_rate) {
    burner = max_rate;
  }
  *tank_temperature +=
      (burner - coil_heat_w - loss) * duration_s / capacity;
  return burner;
}

double cenv_hvac_water_flow_rate(double differential_pressure,
                                 double flow_factor,
                                 double header_resistance) {
  const double resistance =
      1.0 + flow_factor * flow_factor * header_resistance;
  return flow_factor * sqrt(differential_pressure / resistance) / 3600.0;
}

double cenv_hvac_pump_power(double flow, double head,
                            double water_density, double gravity,
                            double efficiency) {
  return flow * water_density * gravity * head / efficiency;
}

double cenv_hvac_head_to_pressure(double head, double water_density,
                                  double gravity, double pascals_per_bar) {
  return gravity * water_density * head / pascals_per_bar;
}

double cenv_hvac_pressure_to_head(double pressure, double water_density,
                                  double gravity, double pascals_per_bar) {
  return pressure / (gravity * water_density / pascals_per_bar);
}
