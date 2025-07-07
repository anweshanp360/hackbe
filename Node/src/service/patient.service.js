const { sql, poolPromise } = require('../db/config');

const getPatients = async (filters) => {
  let query = `SELECT * FROM Patients WHERE 1=1`;
  const inputs = [];

  // Required fields
  if (filters.patient_name) {
    query += ` AND patient_name LIKE @patient_name`;
    inputs.push({ name: 'patient_name', type: sql.NVarChar, value: `%${filters.patient_name}%` });
  }
  if (filters.age) {
    query += ` AND age = @age`;
    inputs.push({ name: 'age', type: sql.Int, value: parseInt(filters.age) });
  }
  if (filters.gender) {
    query += ` AND gender = @gender`;
    inputs.push({ name: 'gender', type: sql.VarChar, value: filters.gender });
  }
  if (filters.symptom_duration) {
    query += ` AND symptom_duration = @symptom_duration`;
    inputs.push({ name: 'symptom_duration', type: sql.Int, value: parseInt(filters.symptom_duration) });
  }

  // Boolean (BIT) fields
  const booleanFields = [
    'muscle_weakness', 'twitching', 'speech_difficulty', 'swallowing_difficulty',
    'breathing_difficulty', 'family_history'
  ];
  booleanFields.forEach(field => {
    if (filters[field] !== undefined) {
      query += ` AND ${field} = @${field}`;
      inputs.push({ name: field, type: sql.Bit, value: filters[field] === 'true' || filters[field] === true ? 1 : 0 });
    }
  });

  // Optional dropdown/text fields
  const optionalFields = [
    { name: 'previous_diagnosis', type: sql.NVarChar },
    { name: 'current_treatment', type: sql.VarChar },
    { name: 'biomarker_status', type: sql.VarChar },
    { name: 'EMG_result', type: sql.VarChar }
  ];
  optionalFields.forEach(field => {
    if (filters[field.name]) {
      query += ` AND ${field.name} = @${field.name}`;
      inputs.push({ name: field.name, type: field.type, value: filters[field.name] });
    }
  });

  try {
    const pool = await poolPromise;
    const request = pool.request();

    inputs.forEach(input => {
      request.input(input.name, input.type, input.value);
    });

    const result = await request.query(query);
    return result.recordset;
  } catch (err) {
    console.error('❌ Error in patient.service:', err);
    throw err;
  } 
};

const postPatient = async (patientData) => {
  const {
    patient_name,
    age,
    gender,
    symptom_duration,
    muscle_weakness,
    twitching,
    speech_difficulty,
    swallowing_difficulty,
    breathing_difficulty,
    family_history,
    previous_diagnosis,
    current_treatment,
    biomarker_status,
    EMG_result
  } = patientData;

  const query = `
    INSERT INTO Patients (
      patient_name, age, gender, symptom_duration,
      muscle_weakness, twitching, speech_difficulty, swallowing_difficulty,
      breathing_difficulty, family_history,
      previous_diagnosis, current_treatment, biomarker_status, EMG_result
    )
    VALUES (
      @patient_name, @age, @gender, @symptom_duration,
      @muscle_weakness, @twitching, @speech_difficulty, @swallowing_difficulty,
      @breathing_difficulty, @family_history,
      @previous_diagnosis, @current_treatment, @biomarker_status, @EMG_result
    )
  `;

  try {
    const pool = await poolPromise;
    const request = pool.request();

    request.input('patient_name', sql.NVarChar, patient_name);
    request.input('age', sql.Int, parseInt(age));
    request.input('gender', sql.VarChar, gender);
    request.input('symptom_duration', sql.Int, parseInt(symptom_duration));

    request.input('muscle_weakness', sql.Bit, muscle_weakness ? 1 : 0);
    request.input('twitching', sql.Bit, twitching ? 1 : 0);
    request.input('speech_difficulty', sql.Bit, speech_difficulty ? 1 : 0);
    request.input('swallowing_difficulty', sql.Bit, swallowing_difficulty ? 1 : 0);
    request.input('breathing_difficulty', sql.Bit, breathing_difficulty ? 1 : 0);
    request.input('family_history', sql.Bit, family_history ? 1 : 0);

    request.input('previous_diagnosis', sql.NVarChar, previous_diagnosis || null);
    request.input('current_treatment', sql.VarChar, current_treatment || null);
    request.input('biomarker_status', sql.VarChar, biomarker_status || null);
    request.input('EMG_result', sql.VarChar, EMG_result || null);

    await request.query(query);
    return { success: true, message: 'Patient data inserted successfully' };
  } catch (err) {
    console.error('❌ Error in postPatient.service:', err);
    throw err;
  }
};

module.exports = {
  getPatients,
  postPatient
};
