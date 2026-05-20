
CREATE CONSTRAINT case_primaryid IF NOT EXISTS
FOR (c:Case) REQUIRE c.primaryid IS UNIQUE;

CREATE CONSTRAINT event_date_value IF NOT EXISTS
FOR (n:EventDate) REQUIRE n.value IS UNIQUE;

CREATE CONSTRAINT report_date_value IF NOT EXISTS
FOR (n:ReportDate) REQUIRE n.value IS UNIQUE;

CREATE CONSTRAINT age_value IF NOT EXISTS
FOR (n:Age) REQUIRE n.value IS UNIQUE;

CREATE CONSTRAINT gender_value IF NOT EXISTS
FOR (n:Gender) REQUIRE n.value IS UNIQUE;

CREATE CONSTRAINT manufacturer_name IF NOT EXISTS
FOR (m:Manufacturer) REQUIRE m.manufacturerName IS UNIQUE;
CREATE CONSTRAINT drug_name_unique IF NOT EXISTS
FOR (d:Drug) REQUIRE d.name IS UNIQUE;

CREATE CONSTRAINT drug_role_unique IF NOT EXISTS
FOR (r:DrugRole) REQUIRE r.value IS UNIQUE;

CREATE CONSTRAINT drug_route_unique IF NOT EXISTS
FOR (r:DrugRoute) REQUIRE r.value IS UNIQUE;

CREATE CONSTRAINT dose_unit_unique IF NOT EXISTS
FOR (u:DoseUnit) REQUIRE u.value IS UNIQUE;

CREATE CONSTRAINT indication_unique IF NOT EXISTS
FOR (i:Indication) REQUIRE i.value IS UNIQUE;

// indexes
CREATE INDEX index_case_age IF NOT EXISTS 
FOR (n:Case) ON (n.age);

CREATE INDEX index_case_ageUnit IF NOT EXISTS 
FOR (n:Case) ON (n.ageUnit);

CREATE INDEX index_case_gender IF NOT EXISTS 
FOR (n:Case) ON (n.gender);

CREATE INDEX index_case_eventdate IF NOT EXISTS 
FOR (n:Case) ON (n.eventDate);

CREATE INDEX index_case_reportdate IF NOT EXISTS 
FOR (n:Case) ON (n.reportDate);


LOAD CSV WITH HEADERS FROM "https://raw.githubusercontent.com/neo4j-graph-examples/healthcare-analytics/main/data/csv/demographics.csv" AS row

MERGE (c:Case { primaryid: toInteger(row.primaryid) })

WITH row, c

MERGE (eventDate:EventDate { value: date(row.eventDate) })
MERGE (c)-[:HAS_EVENT_DATE]->(eventDate)

MERGE (reportDate:ReportDate { value: date(row.reportDate) })
MERGE (c)-[:HAS_REPORT_DATE]->(reportDate)

MERGE (age:Age { value: toFloat(row.age) })
MERGE (c)-[:HAS_AGE]->(age)

MERGE (ageUnit:AgeUnit { value: row.ageUnit })
MERGE (c)-[:HAS_AGE_UNIT]->(ageUnit)

MERGE (gender:Gender { value: row.sex })
MERGE (c)-[:HAS_GENDER]->(gender)

MERGE (occupation:ReporterOccupation { value: row.reporterOccupation })
MERGE (c)-[:HAS_REPORTER_OCCUPATION]->(occupation)

MERGE (m:Manufacturer { manufacturerName: row.manufacturerName })
MERGE (m)-[:REGISTERED]->(c)

MERGE (ageGroup:AgeGroup { ageGroup: row.ageGroup })
MERGE (c)-[:FALLS_UNDER]->(ageGroup)

RETURN count(c);


LOAD CSV WITH HEADERS FROM "https://raw.githubusercontent.com/neo4j-graph-examples/healthcare-analytics/main/data/csv/outcome.csv" AS row

// Conditionally create outcome node
MERGE (o:Outcome { code: row.code })
ON CREATE SET
o.outcome = row.outcome

WITH o, row

// Find the case to relate this outcome to
MATCH (c:Case {primaryid: toInteger(row.primaryid)})

// Relate
MERGE (c)-[:RESULTED_IN]->(o)

RETURN count(o);


LOAD CSV WITH HEADERS FROM "https://raw.githubusercontent.com/neo4j-graph-examples/healthcare-analytics/main/data/csv/reaction.csv" AS row

// Find case
MATCH (c:Case {primaryid: toInteger(row.primaryid)})

// Create one Reaction occurrence per case+description pair
MERGE (r:Reaction {
  description: row.description
})

// Connect case to reaction
MERGE (c)-[:HAS_REACTION]->(r)

// Reify description as shared node (deduplicated)
MERGE (d:ReactionDescription { value: row.description })

// Connect reaction to description
MERGE (r)-[:HAS_DESCRIPTION]->(d)

RETURN count(r);


LOAD CSV WITH HEADERS FROM "https://raw.githubusercontent.com/neo4j-graph-examples/healthcare-analytics/main/data/csv/reportSources.csv" AS row

// Conditionally create reportSource node
MERGE (r:ReportSource { code: row.code })
ON CREATE SET
r.name = row.name

WITH r, row

// Find the case to relate this report source to
MATCH (c:Case {primaryid: toInteger(row.primaryid) })

WITH c, r

// Relate
MERGE (c)-[:REPORTED_BY]->(r)

RETURN count(r);



CALL {
  WITH "https://raw.githubusercontent.com/neo4j-graph-examples/healthcare-analytics/main/data/csv/drugs-indication.csv" AS url
  LOAD CSV WITH HEADERS FROM url AS row

  MATCH (c:Case {primaryid: toInteger(row.primaryid)})

  MERGE (d:Drug {name: row.name})
    ON CREATE SET d.primarySubstance = row.primarySubstabce

  MERGE (e:DrugExposure {
    primaryid: toInteger(row.primaryid),
    drugName: row.name,
    drugSequence: row.drugSequence
  })

  MERGE (c)-[:HAS_DRUG_EXPOSURE]->(e)
  MERGE (e)-[:INVOLVES_DRUG]->(d)

  MERGE (role:DrugRole {value: row.role})
  MERGE (e)-[:HAS_ROLE]->(role)

  MERGE (seq:DrugSequence {value: row.drugSequence})
  MERGE (e)-[:HAS_DRUG_SEQUENCE]->(seq)

  MERGE (route:DrugRoute {value: row.route})
  MERGE (e)-[:HAS_ROUTE]->(route)

  MERGE (doseAmount:DoseAmount {value: row.doseAmount})
  MERGE (e)-[:HAS_DOSE_AMOUNT]->(doseAmount)

  MERGE (doseUnit:DoseUnit {value: row.doseUnit})
  MERGE (e)-[:HAS_DOSE_UNIT]->(doseUnit)

  MERGE (indication:Indication {value: row.indication})
  MERGE (e)-[:HAS_INDICATION]->(indication)

} IN TRANSACTIONS OF 500 ROWS;

LOAD CSV WITH HEADERS FROM "https://raw.githubusercontent.com/neo4j-graph-examples/healthcare-analytics/main/data/csv/therapy.csv" AS row

MATCH (c:Case {primaryid: toInteger(row.primaryid)})
MATCH (d:Drug {name: row.drugName})

MERGE (t:Therapy {
  primaryid: toInteger(row.primaryid)
})

MERGE (c)-[:RECEIVED]->(t)

MERGE (p:Prescription {
  primaryid: toInteger(row.primaryid),
  drugName: row.drugName,
  drugSequence: row.drugSequence
})

MERGE (t)-[:HAS_PRESCRIPTION]->(p)
MERGE (p)-[:PRESCRIBED_DRUG]->(d)

MERGE (seq:DrugSequence {value: row.drugSequence})
MERGE (p)-[:HAS_DRUG_SEQUENCE]->(seq)

MERGE (startYear:StartYear {value: toInteger(coalesce(row.startYear, "1900"))})
MERGE (p)-[:HAS_START_YEAR]->(startYear)

MERGE (endYear:EndYear {value: toInteger(coalesce(row.endYear, "2021"))})
MERGE (p)-[:HAS_END_YEAR]->(endYear);

//For us 

MATCH (n:AgeGroup) set n.ageGroup_GT = n.ageGroup;
MATCH (n:Drug) set n.name_GT = n.name, n.primarySubstance_GT=n.primarySubstance;
MATCH (n:Manufacturer) set n.manufacturerName_GT = n.manufacturerName;
MATCH (n:Outcome) set n.outcome_GT = n.outcome;
MATCH (n:Reaction) set n.description_GT = n.description;
MATCH (n:ReportSource) set n.name_GT = n.name;
MATCH (n:Therapy) set n.primaryid_GT = n.primaryid;
MATCH (n:Case) set n.reporterOccupation_GT = n.reporterOccupation , n.gender_GT = n.gender ,n.age_GT = n.age;



match ()-[r]->() set r.toDelete=False;

MATCH (n:AgeGroup {ageGroup:"Infant"}) set n.minAge=0,n.maxAge=1; MATCH (n:AgeGroup {ageGroup:"Child"}) set n.minAge=2,n.maxAge=12; MATCH (n:AgeGroup {ageGroup:"Adolescent"}) set n.minAge=13,n.maxAge=18; MATCH (n:AgeGroup {ageGroup:"Adult"}) set n.minAge=19,n.maxAge=64; MATCH (n:AgeGroup {ageGroup:"Elderly"}) set n.minAge=65,n.maxAge=120;


// MATCH (c:Case)-[rel:HAS_DRUG_EXPOSURE]-(d:DrugExposure)-[rel1:HAS_ROLE]-(r:DrugRole),(d)-[rel4:HAS_DOSE_AMOUNT]-(a:DoseAmount), (c)-[rel3:HAS_DRUG_EXPOSURE]-(d1:DrugExposure)-[rel2:HAS_ROLE]-(r1:DrugRole),(d1)-[rel5:HAS_DOSE_AMOUNT]-(a1:DoseAmount) where r.value="Primary Suspect" and r1.value="Secondary Suspect"  and d.drugName=d1.drugName and d<>d1 and a.value=a1.value return count(c);

CALL apoc.periodic.iterate(
"
MATCH (c:Case)-[rel:HAS_DRUG_EXPOSURE]-(d:DrugExposure)-[rel1:HAS_ROLE]-(r:DrugRole),
      (d)-[rel4:HAS_DOSE_AMOUNT]-(a:DoseAmount),
      (c)-[rel3:HAS_DRUG_EXPOSURE]-(d1:DrugExposure)-[rel2:HAS_ROLE]-(r1:DrugRole),
      (d1)-[rel5:HAS_DOSE_AMOUNT]-(a1:DoseAmount)
WHERE r.value = 'Primary Suspect'
  AND r1.value = 'Secondary Suspect'
  AND d.drugName = d1.drugName
  AND d <> d1
  AND a.value = a1.value
RETURN DISTINCT c,d,d1,r,r1,a,a1,rel,rel1,rel2,rel3,rel4,rel5
",
"
WITH c,d,d1,r,r1,a,a1,rel,rel1,rel2,rel3,rel4,rel5,
     apoc.create.uuid() AS uuid,
     rand() AS randomValue,
     toString(toFloat(a.value) * 2) AS newSecondaryDoseAmount

SET c.isViolation = true,
    d.isViolation = true,
    d1.isViolation = true,
    r.isViolation = true,
    r1.isViolation = true,
    a.isViolation = true,
    a1.isViolation = true,
    rel.isViolation = true,
    rel1.isViolation = true,
    rel2.isViolation = true,
    rel3.isViolation = true,
    rel4.isViolation = true,
    rel5.isViolation = true,

    c.violationId = coalesce(c.violationId, []) + uuid,
    d.violationId = coalesce(d.violationId, []) + uuid,
    d1.violationId = coalesce(d1.violationId, []) + uuid,
    r.violationId = coalesce(r.violationId, []) + uuid,
    r1.violationId = coalesce(r1.violationId, []) + uuid,
    a.violationId = coalesce(a.violationId, []) + uuid,
    a1.violationId = coalesce(a1.violationId, []) + uuid,
    rel.violationId = coalesce(rel.violationId, []) + uuid,
    rel1.violationId = coalesce(rel1.violationId, []) + uuid,
    rel2.violationId = coalesce(rel2.violationId, []) + uuid,
    rel3.violationId = coalesce(rel3.violationId, []) + uuid,
    rel4.violationId = coalesce(rel4.violationId, []) + uuid,
    rel5.violationId = coalesce(rel5.violationId, []) + uuid,
    rel.violationType=1,
    rel1.violationType=1,
    rel2.violationType=1,
    rel3.violationType=1,
    rel4.violationType=1,
    rel5.violationType=1,

    a1.value_GT = CASE
      WHEN randomValue < 0.5 THEN newSecondaryDoseAmount
      ELSE a1.value
    END,

    rel3.toDelete_GT = CASE
      WHEN randomValue >= 0.5 THEN true
      ELSE false
    END,

    rel3.toDelete = CASE
      WHEN randomValue >= 0.5 THEN true
      ELSE false
    END

CREATE (:Violation {
  violationId: uuid,

  repairs:
  '[MATCH (a1:DoseAmount) WHERE elementId(a1)=$a1 SET a1.value=$newSecondaryDoseAmount,
    MATCH (:Case)-[rel3:HAS_DRUG_EXPOSURE]-(:DrugExposure) WHERE elementId(rel3)=$rel3 SET rel3.deleted=true,
    MATCH (:Case)-[rel:HAS_DRUG_EXPOSURE]-(:DrugExposure) WHERE elementId(rel)=$rel SET rel.deleted=true,
    MATCH (:Case)-[rel:HAS_DRUG_EXPOSURE]-(:DrugExposure), (a:DoseAmount) WHERE elementId(rel)=$rel AND elementId(a)=$a SET rel.deleted=true, a.value=$newSecondaryDoseAmount,
    MATCH (:Case)-[rel3:HAS_DRUG_EXPOSURE]-(:DrugExposure), (a1:DoseAmount) WHERE elementId(rel3)=$rel3 AND elementId(a1)=$a1 SET rel3.deleted=true, a1.value=$newSecondaryDoseAmount]',

  order: CASE
    WHEN randomValue < 0.5 THEN [0,2,1]
    ELSE [1,2,0]
  END,

  repair_GT: CASE
    WHEN randomValue < 0.5 THEN 0
    ELSE 1
  END,

  newSecondaryDoseAmount_GT: newSecondaryDoseAmount,
  violationType: 1
})
",
{
  batchSize: 100,
  parallel: false
});

//MATCH (a:Age)-[rel:HAS_AGE]-(c:Case)-[rel1:FALLS_UNDER]->(ag:AgeGroup) where a.value<ag.minAge or a.value>ag.maxAge return c,a.value,ag;
CALL apoc.periodic.iterate(
"
MATCH (a:Age)-[rel:HAS_AGE]-(c:Case)-[rel1:FALLS_UNDER]->(ag:AgeGroup)
WHERE toInteger(a.value) < ag.minAge
   OR toInteger(a.value) > ag.maxAge
RETURN DISTINCT c,a,ag,rel,rel1
",
"
WITH c,a,ag,rel,rel1,
     apoc.create.uuid() AS uuid,
     rand() AS randomValue,
     toString(ag.minAge) AS newAgeValue

SET c.isViolation = true,
    a.isViolation = true,
    ag.isViolation = true,
    rel.isViolation = true,
    rel1.isViolation = true,

    c.violationId = coalesce(c.violationId, []) + uuid,
    a.violationId = coalesce(a.violationId, []) + uuid,
    ag.violationId = coalesce(ag.violationId, []) + uuid,
    rel.violationId = coalesce(rel.violationId, []) + uuid,
    rel1.violationId = coalesce(rel1.violationId, []) + uuid,
    rel.violationType = 2,
    rel1.violationType = 2,

    a.value_GT = CASE
      WHEN randomValue < 0.5 THEN newAgeValue
      ELSE a.value
    END,

    rel1.toDelete_GT = CASE
      WHEN randomValue >= 0.5 THEN true
      ELSE false
    END,

    rel1.toDelete = CASE
      WHEN randomValue >= 0.5 THEN true
      ELSE false
    END

CREATE (:Violation {
  violationId: uuid,

  repairs:
  '[MATCH (a:Age) WHERE elementId(a)=$a SET a.value=$newAgeValue,
    MATCH (:Case)-[rel1:FALLS_UNDER]->(:AgeGroup) WHERE elementId(rel1)=$rel1 SET rel1.deleted=true,
    MATCH (a:Age), (:Case)-[rel1:FALLS_UNDER]->(:AgeGroup) WHERE elementId(a)=$a AND elementId(rel1)=$rel1 SET a.value=$newAgeValue, rel1.deleted=true]',

  order: CASE
    WHEN randomValue < 0.5 THEN [0,2,1]
    ELSE [1,2,0]
  END,

  repair_GT: CASE
    WHEN randomValue < 0.5 THEN 0
    ELSE 1
  END,

  newAgeValue_GT: newAgeValue,
  violationType: 2
})
",
{
  batchSize: 100,
  parallel: false
});

// MATCH (c:Case)-[rel:HAS_EVENT_DATE]-(d:EventDate),(c)-[rel1:HAS_AGE]-(a:Age),(c)-[rel2:HAS_GENDER]-(g:Gender),(c1:Case)-[rel3:HAS_EVENT_DATE]-(d:EventDate),(c1)-[rel4:HAS_AGE]-(a),(c1)-[rel5:HAS_GENDER]-(g)  where c<>c1 return c,c1;


CALL apoc.periodic.iterate(
"
MATCH (c:Case)-[rel:HAS_EVENT_DATE]-(d:EventDate),
      (c)-[rel1:HAS_AGE]-(a:Age),
      (c)-[rel2:HAS_GENDER]-(g:Gender),
      (c1:Case)-[rel3:HAS_EVENT_DATE]-(d),
      (c1)-[rel4:HAS_AGE]-(a),
      (c1)-[rel5:HAS_GENDER]-(g)
WHERE c <> c1
RETURN DISTINCT c,c1,d,a,g,rel,rel1,rel2,rel3,rel4,rel5
",
"
WITH c,c1,d,a,g,rel,rel1,rel2,rel3,rel4,rel5,
     apoc.create.uuid() AS uuid,
     rand() AS randomValue,
     toString(date(d.value) + duration({days: 1})) AS newEventDateValue

SET c.isViolation = true,
    c1.isViolation = true,
    d.isViolation = true,
    a.isViolation = true,
    g.isViolation = true,
    rel.isViolation = true,
    rel1.isViolation = true,
    rel2.isViolation = true,
    rel3.isViolation = true,
    rel4.isViolation = true,
    rel5.isViolation = true,

    c.violationId = coalesce(c.violationId, []) + uuid,
    c1.violationId = coalesce(c1.violationId, []) + uuid,
    d.violationId = coalesce(d.violationId, []) + uuid,
    a.violationId = coalesce(a.violationId, []) + uuid,
    g.violationId = coalesce(g.violationId, []) + uuid,
    rel.violationId = coalesce(rel.violationId, []) + uuid,
    rel1.violationId = coalesce(rel1.violationId, []) + uuid,
    rel2.violationId = coalesce(rel2.violationId, []) + uuid,
    rel3.violationId = coalesce(rel3.violationId, []) + uuid,
    rel4.violationId = coalesce(rel4.violationId, []) + uuid,
    rel5.violationId = coalesce(rel5.violationId, []) + uuid,
    rel.violationType = 3,
    rel1.violationType = 3,
    rel2.violationType = 3,
    rel3.violationType = 3,
    rel4.violationType = 3,
    rel5.violationType = 3,

    d.value_GT = CASE
      WHEN randomValue < 0.5 THEN d.value
      ELSE newEventDateValue
    END,

    rel3.toDelete_GT = CASE
      WHEN randomValue < 0.5 THEN true
      ELSE false
    END,

    rel3.toDelete = CASE
      WHEN randomValue < 0.5 THEN true
      ELSE false
    END

CREATE (:Violation {
  violationId: uuid,

  repairs:
  '[MATCH (:Case)-[rel3:HAS_EVENT_DATE]-(:EventDate) WHERE elementId(rel3)=$rel3 SET rel3.deleted=true,
    MATCH (d:EventDate) WHERE elementId(d)=$d SET d.value=$newEventDateValue,
    MATCH (:Case)-[rel3:HAS_EVENT_DATE]-(:EventDate), (d:EventDate) WHERE elementId(rel3)=$rel3 AND elementId(d)=$d SET rel3.deleted=true, d.value=$newEventDateValue]',

  order: CASE
    WHEN randomValue < 0.5 THEN [0,2,1]
    ELSE [1,2,0]
  END,

  repair_GT: CASE
    WHEN randomValue < 0.5 THEN 0
    ELSE 1
  END,

  newEventDateValue_GT: newEventDateValue,
  violationType: 3
})
",
{
  batchSize: 100,
  parallel: false
});




