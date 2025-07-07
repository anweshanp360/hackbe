const { poolPromise } = require("../db/config");

const getAllTrials = async () => {
  try {
    const pool = await poolPromise;
    const result = await pool.request().query("SELECT * FROM dbo.Trials");

    return result.recordset;
  } catch (err) {
    throw new Error("Failed to fetch trials: " + err.message);
  }
};

const createTrial = async (trialData) => {
  try {
    const pool = await sql.connect(config);
    const request = pool.request();

    request.input('trial_name', sql.VarChar, trialData.trial_name);
    request.input('sponsor', sql.VarChar, trialData.sponsor);
    request.input('min_age', sql.Int, trialData.min_age);
    request.input('max_age', sql.Int, trialData.max_age);
    request.input('gender_requirement', sql.VarChar, trialData.gender_requirement);
    request.input('min_symptom_duration', sql.Int, trialData.min_symptom_duration);
    request.input('requires_muscle_weakness', sql.Bit, trialData.requires_muscle_weakness);
    request.input('requires_twitching', sql.Bit, trialData.requires_twitching);
    request.input('requires_positive_biomarker', sql.VarChar, trialData.requires_positive_biomarker);
    request.input('requires_abnormal_emg', sql.VarChar, trialData.requires_abnormal_emg);
    request.input('allowed_treatments', sql.VarChar, trialData.allowed_treatments);
    request.input('exclusion_previous_diagnosis', sql.VarChar, trialData.exclusion_previous_diagnosis);
    request.input('location', sql.VarChar, trialData.location);
    request.input('status', sql.VarChar, trialData.status);
    request.input('start_date', sql.Date, trialData.start_date);
    request.input('end_date', sql.Date, trialData.end_date);

    const result = await request.query(`
      INSERT INTO Trials (
        trial_name, sponsor, min_age, max_age, gender_requirement,
        min_symptom_duration, requires_muscle_weakness, requires_twitching,
        requires_positive_biomarker, requires_abnormal_emg, allowed_treatments,
        exclusion_previous_diagnosis, location, status, start_date, end_date
      ) VALUES (
        @trial_name, @sponsor, @min_age, @max_age, @gender_requirement,
        @min_symptom_duration, @requires_muscle_weakness, @requires_twitching,
        @requires_positive_biomarker, @requires_abnormal_emg, @allowed_treatments,
        @exclusion_previous_diagnosis, @location, @status, @start_date, @end_date
      );
    `);

    return { success: true, rowsAffected: result.rowsAffected };
  } catch (error) {
    console.error('Error creating trial:', error);
    throw error;
  }
};

module.exports = { getAllTrials,createTrial};
