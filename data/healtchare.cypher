

// Constraints
CREATE CONSTRAINT constraint_drug_name IF NOT EXISTS 
FOR (n:Drug) REQUIRE n.name IS UNIQUE;

CREATE CONSTRAINT constraint_case_primaryid IF NOT EXISTS 
FOR (n:Case) REQUIRE n.primaryid IS UNIQUE;

CREATE CONSTRAINT constraint_reaction_description IF NOT EXISTS 
FOR (n:Reaction) REQUIRE n.description IS UNIQUE;

CREATE CONSTRAINT constraint_reportsource_code IF NOT EXISTS 
FOR (n:ReportSource) REQUIRE n.code IS UNIQUE;

CREATE CONSTRAINT constraint_outcome_code IF NOT EXISTS 
FOR (n:Outcome) REQUIRE n.code IS UNIQUE;

CREATE CONSTRAINT constraint_therapy_primaryid IF NOT EXISTS 
FOR (n:Therapy) REQUIRE n.primaryid IS UNIQUE;

CREATE CONSTRAINT constraint_manufacturer_name IF NOT EXISTS 
FOR (n:Manufacturer) REQUIRE n.manufacturerName IS UNIQUE;


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

//Conditionally create Case nodes, set properties on first create
MERGE (c:Case { primaryid: toInteger(row.primaryid) })
ON CREATE SET
c.eventDate= date(row.eventDate),
c.reportDate= date(row.reportDate),
c.age = toFloat(row.age),
c.ageUnit = row.ageUnit,
c.gender = row.sex,
c.reporterOccupation = row.reporterOccupation

//Conditionally create Manufacturer
MERGE (m:Manufacturer { manufacturerName: row.manufacturerName } )

//Relate case and manufacturer
MERGE (m)-[:REGISTERED]->(c)

//Conditionally create age group node and relate with case
MERGE (a:AgeGroup { ageGroup: row.ageGroup })

//Relate case with age group
MERGE (c)-[:FALLS_UNDER]->(a)

RETURN count (c);


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

//Conditionally create reaction node
MERGE (r:Reaction { description: row.description })

WITH r, row

//Find the case to relate this reaction to
MATCH (c:Case {primaryid: toInteger(row.primaryid)})

//Relate
MERGE (c)-[:HAS_REACTION]->(r)

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

  WITH row
  MERGE (d:Drug {name: row.name})
    ON CREATE SET d.primarySubstance = row.primarySubstabce

  WITH d, row
  MATCH (c:Case {primaryid: toInteger(row.primaryid)})

  FOREACH (_ IN CASE WHEN row.role = "Primary Suspect" THEN [1] ELSE [] END |
    MERGE (c)-[:IS_PRIMARY_SUSPECT {
      drugSequence: row.drugSequence,
      route: row.route,
      doseAmount: row.doseAmount,
      doseUnit: row.doseUnit,
      indication: row.indication
    }]->(d)
  )

  FOREACH (_ IN CASE WHEN row.role = "Secondary Suspect" THEN [1] ELSE [] END |
    MERGE (c)-[:IS_SECONDARY_SUSPECT {
      drugSequence: row.drugSequence,
      route: row.route,
      doseAmount: row.doseAmount,
      doseUnit: row.doseUnit,
      indication: row.indication
    }]->(d)
  )

  FOREACH (_ IN CASE WHEN row.role = "Concomitant" THEN [1] ELSE [] END |
    MERGE (c)-[:IS_CONCOMITANT {
      drugSequence: row.drugSequence,
      route: row.route,
      doseAmount: row.doseAmount,
      doseUnit: row.doseUnit,
      indication: row.indication
    }]->(d)
  )

  FOREACH (_ IN CASE WHEN row.role = "Interacting" THEN [1] ELSE [] END |
    MERGE (c)-[:IS_INTERACTING {
      drugSequence: row.drugSequence,
      route: row.route,
      doseAmount: row.doseAmount,
      doseUnit: row.doseUnit,
      indication: row.indication
    }]->(d)
  )
} IN TRANSACTIONS OF 500 ROWS;


LOAD CSV WITH HEADERS FROM "https://raw.githubusercontent.com/neo4j-graph-examples/healthcare-analytics/main/data/csv/therapy.csv" AS row

//Conditionally create therapy node
MERGE (t:Therapy { primaryid: toInteger(row.primaryid) })

WITH t, row

//Find the case to relate this therapy to
MATCH (c:Case {primaryid: toInteger(row.primaryid)})

//Relate case with therapy
MERGE (c)-[:RECEIVED]->(t)

WITH c, t, row

//Find drugs prescribed in the therapy
MATCH (d:Drug { name: row.drugName })

//Relate therapy and drugs
MERGE (t)-[:PRESCRIBED { drugSequence: row.drugSequence, startYear: coalesce(row.startYear, 1900), endYear: coalesce(row.endYear, 2021) } ]->(d);




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





CALL apoc.periodic.iterate(
  "
  MATCH (c:Case)-[primary:IS_PRIMARY_SUSPECT]->(d:Drug)<-[secondary:IS_SECONDARY_SUSPECT]-(c)
  RETURN DISTINCT c, d, primary, secondary
  ",
  "
  WITH c, d, primary, secondary, apoc.create.uuid() AS uuid
  SET c.isViolation = true,
      d.isViolation = true,
      primary.isViolation = true,
      secondary.isViolation = true,

      c.violationId = coalesce(c.violationId, []) + uuid,
      d.violationId = coalesce(d.violationId, []) + uuid,
      primary.violationId = coalesce(primary.violationId, []) + uuid,
      secondary.violationId = coalesce(secondary.violationId, []) + uuid,
      secondary.toDelete = true

  CREATE (v:Violation {
      violationId: uuid,
      repairs: '[MATCH ()-[r:IS_PRIMARY_SUSPECT]-() WHERE r.elementId=$primary SET r.deleted=true,MATCH ()-[p:IS_SECONDARY_SUSPECT]-() WHERE p.elementId=$secondary SET p.deleted=true,MATCH ()-[r:IS_PRIMARY_SUSPECT]-()-[p:IS_SECONDARY_SUSPECT]-() WHERE r.elementId=$primary AND p.elementId=$secondary SET r.deleted=true, p.deleted=true]',
      order: [1,2,0],
      
  })
  ",
  {
    batchSize: 100,
    parallel: false
  }
);



MATCH (d:Drug)
WITH d LIMIT 200
WITH d, apoc.create.uuid() AS uuid
CREATE (d)-[r:IS_PRIMARY_SUSPECT]->(d)
SET d.isViolation = true,
    d.violationId = coalesce(d.violationId, []) + uuid,
    r.isViolation = true,
    r.toDelete = true,
    r.violationId = coalesce(r.violationId, []) + uuid
CREATE (v:Violation {violationId: uuid,
          repairs: '[MATCH ()-[r:IS_PRIMARY_SUSPECT]-() WHERE r.elementId=" + primaryId + " SET r.deleted=true]',
          order: [0]
  })
    ;



MATCH (a1:AgeGroup), (a2:AgeGroup)
WHERE a1 <> a2
WITH a1, a2
LIMIT 1

// Pick 200 cases with only one FALLS_UNDER relationship
MATCH (c:Case)-[:FALLS_UNDER]->(a1)
WHERE NOT (c)-[:FALLS_UNDER]->(a2)
WITH c, a2
LIMIT 200

// Inject the violation
MERGE (c)-[:FALLS_UNDER]->(a2);


CALL apoc.periodic.iterate(
  "
  MATCH (c:Case)-[r1:FALLS_UNDER]->(a1:AgeGroup),
        (c)-[r2:FALLS_UNDER]->(a2:AgeGroup)
  WHERE a1 <> a2
  RETURN DISTINCT c, a1, a2, r1, r2
  ",
  "
  WITH c, a1, a2, r1, r2, apoc.create.uuid() AS uuid, rand() AS randomValue

  SET c.isViolation = true,
      a1.isViolation = true,
      a2.isViolation = true,
      r1.isViolation = true,
      r2.isViolation = true,

      c.violationId = coalesce(c.violationId, []) + uuid,
      a1.violationId = coalesce(a1.violationId, []) + uuid,
      a2.violationId = coalesce(a2.violationId, []) + uuid,
      r1.violationId = coalesce(r1.violationId, []) + uuid,
      r2.violationId = coalesce(r2.violationId, []) + uuid

  FOREACH (_ IN CASE WHEN randomValue < 0.5 THEN [1] ELSE [] END |
    SET r1.toDelete=True
    CREATE (v:Violation {violationId: uuid,
           repairs: '[MATCH ()-[r:FALLS_UNDER]-() WHERE r.elementId=$r1 SET r.deleted=true,MATCH ()-[p:FALLS_UNDER]-() WHERE p.elementId=$r2 SET p.deleted=true,MATCH ()-[r:FALLS_UNDER]-()-[p:FALLS_UNDER]-() WHERE r.elementId=$r1 AND p.elementId=$r2 SET r.deleted=true, p.deleted=true]',
          order : [0,2,1]
  })
  )
  FOREACH (_ IN CASE WHEN randomValue >= 0.5 THEN [1] ELSE [] END |
    SET r2.toDelete=True
    CREATE (v:Violation {violationId: uuid,
           repairs: '[MATCH ()-[r:FALLS_UNDER]-() WHERE r.elementId=$r1 SET r.deleted=true,MATCH ()-[p:FALLS_UNDER]-() WHERE p.elementId=$r2 SET p.deleted=true,MATCH ()-[r:FALLS_UNDER]-()-[p:FALLS_UNDER]-() WHERE r.elementId=$r1 AND p.elementId=$r2 SET r.deleted=true, p.deleted=true]',
          order : '[1,2,0]'
  })
  )
  
  ",
  {
    batchSize: 100,
    parallel: false
  }
); 

CALL apoc.periodic.iterate(
  "
  MATCH (d:Drug)-[p:PRESCRIBED]-(t:Therapy)-[r:RECEIVED]-(c:Case)-[f:FALLS_UNDER]->(ag:AgeGroup {ageGroup: 'Child'})
  RETURN DISTINCT d, t, c, ag, p, r, f
  ",
  "
  WITH d, t, c, ag, p, r, f, apoc.create.uuid() AS uuid, rand() AS randomValue
  SET d.isViolation = true,
      t.isViolation = true,
      c.isViolation = true,
      ag.isViolation = true,
      p.isViolation = true,
      r.isViolation = true,
      f.isViolation = true,
      d.violationId = coalesce(d.violationId, []) + uuid,
      t.violationId = coalesce(t.violationId, []) + uuid,
      c.violationId = coalesce(c.violationId, []) + uuid,
      ag.violationId = coalesce(ag.violationId, []) + uuid,
      p.violationId = coalesce(p.violationId, []) + uuid,
      r.violationId = coalesce(r.violationId, []) + uuid,
      f.violationId = coalesce(f.violationId, []) + uuid
  FOREACH (_ IN CASE WHEN randomValue < 0.5 THEN [1] ELSE [] END |
    set f.toDelete=True
    CREATE (v:Violation {
    violationId: uuid,
    repairs: '[MATCH ()-[pr:PRESCRIBED]-() WHERE pr.elementId = $p SET pr.deleted = true,MATCH ()-[re:RECEIVED]-() WHERE re.elementId = $r SET re.deleted = true,MATCH ()-[fl:FALLS_UNDER]->() WHERE fl.elementId = $f SET fl.deleted = true,MATCH ()-[pr:PRESCRIBED]-()-[re:RECEIVED]-() WHERE pr.elementId=$p AND re.elementId=$r SET pr.deleted=true, re.deleted=true,MATCH ()-[pr:PRESCRIBED]-()-[fl:FALLS_UNDER]->() WHERE pr.elementId=$p AND fl.elementId=$f SET pr.deleted=true, fl.deleted=true,MATCH ()-[re:RECEIVED]-()-[fl:FALLS_UNDER]->() WHERE re.elementId=$r AND fl.elementId=$f SET re.deleted=true, fl.deleted=true,MATCH ()-[pr:PRESCRIBED]-()-[re:RECEIVED]-()-[fl:FALLS_UNDER]->() WHERE pr.elementId=$p AND re.elementId=$r AND fl.elementId=$f SET pr.deleted=true, re.deleted=true, fl.deleted=true]',
    order : [2,4,5,6,0,1,3]
})  
  )
  FOREACH (_ IN CASE WHEN randomValue >= 0.5 THEN [1] ELSE [] END |
   set p.toDelete=True
   CREATE (v:Violation {
    violationId: uuid,
    repairs: '[MATCH ()-[pr:PRESCRIBED]-() WHERE pr.elementId = $p SET pr.deleted = true,MATCH ()-[re:RECEIVED]-() WHERE re.elementId = $r SET re.deleted = true,MATCH ()-[fl:FALLS_UNDER]->() WHERE fl.elementId = $f SET fl.deleted = true,MATCH ()-[pr:PRESCRIBED]-()-[re:RECEIVED]-() WHERE pr.elementId=$p AND re.elementId=$r SET pr.deleted=true, re.deleted=true,MATCH ()-[pr:PRESCRIBED]-()-[fl:FALLS_UNDER]->() WHERE pr.elementId=$p AND fl.elementId=$f SET pr.deleted=true, fl.deleted=true,MATCH ()-[re:RECEIVED]-()-[fl:FALLS_UNDER]->() WHERE re.elementId=$r AND fl.elementId=$f SET re.deleted=true, fl.deleted=true,MATCH ()-[pr:PRESCRIBED]-()-[re:RECEIVED]-()-[fl:FALLS_UNDER]->() WHERE pr.elementId=$p AND re.elementId=$r AND fl.elementId=$f SET pr.deleted=true, re.deleted=true, fl.deleted=true]',
    order : [0,3,4,6,1,2,5]
    
})  
  )
  ",
  {
    batchSize: 100,
    parallel: false
  }
)

