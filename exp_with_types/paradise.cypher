LOAD CSV WITH HEADERS FROM 'file:///paradise/address.csv' AS row

WITH row
WHERE row._id IS NOT NULL AND row._id <> ''

MERGE (a:Address {id: row._id})
SET
  a.value = row.address

WITH a, row
WHERE row.country_codes IS NOT NULL AND row.country_codes <> ''

MERGE (c:Country {code: row.country_codes})

MERGE (a)-[:LOCATED_IN]->(c);


LOAD CSV WITH HEADERS FROM 'file:///paradise/entity.csv' AS row

WITH row
WHERE row._id IS NOT NULL AND row._id <> ''

MERGE (e:Entity {id: row._id})
SET
  e.name = row.name,
  e.ibcRUC = row.ibcRUC,
  e.company_type = row.company_type,
  e.incorporation_date = row.incorporation_date,
  e.status = row.status,
  e.type = row.type

// closed_date -> Date node
WITH e, row
WHERE row.closed_date IS NOT NULL AND row.closed_date <> ''
MERGE (d:Date {value: row.closed_date})
MERGE (e)-[:HAS_CLOSED_DATE]->(d)

WITH e, row
WHERE row.country_codes IS NOT NULL AND row.country_codes <> ''
MERGE (c:Country {code: row.country_codes})
MERGE (e)-[:REGISTERED_IN]->(c)

WITH e, row
WHERE row.jurisdiction IS NOT NULL AND row.jurisdiction <> ''
MERGE (j:Jurisdiction {code: row.jurisdiction})
SET j.description = row.jurisdiction_description
MERGE (e)-[:HAS_JURISDICTION]->(j);



LOAD CSV WITH HEADERS FROM 'file:///paradise/intermediary.csv' AS row

WITH row
WHERE row._id IS NOT NULL AND row._id <> ''

MERGE (i:Intermediary {id: row._id})
SET
  i.name = row.name

WITH i, row
WHERE row.country_codes IS NOT NULL AND row.country_codes <> ''
MERGE (c:Country {code: row.country_codes})
MERGE (i)-[:REGISTERED_IN]->(c);



LOAD CSV WITH HEADERS FROM 'file:///paradise/officer.csv' AS row

WITH row
WHERE row._id IS NOT NULL AND row._id <> ''

MERGE (o:Officer {id: row._id})
SET
  o.name = row.name

WITH o, row
WHERE row.country_codes IS NOT NULL AND row.country_codes <> ''
MERGE (c:Country {code: row.country_codes})
MERGE (o)-[:ASSOCIATED_WITH_COUNTRY]->(c);



LOAD CSV WITH HEADERS FROM 'file:///paradise/other.csv' AS row

WITH row
WHERE row._id IS NOT NULL AND row._id <> ''

MERGE (o:Other {id: row._id})
SET
  o.name = row.name,
  o.status = row.status,
  o.note = row.note,
  o.valid_until = row.valid_until

// Country relationship
WITH o, row
WHERE row.country_codes IS NOT NULL AND row.country_codes <> ''
MERGE (c:Country {code: row.country_codes})
MERGE (o)-[:REGISTERED_IN]->(c)

// Jurisdiction node + relationship
WITH o, row
WHERE row.jurisdiction IS NOT NULL AND row.jurisdiction <> ''
MERGE (j:Jurisdiction {code: row.jurisdiction})
SET j.description = row.jurisdiction_description
MERGE (o)-[:HAS_JURISDICTION]->(j);


LOAD CSV WITH HEADERS FROM 'file:///paradise/connected_to.csv' AS row

WITH row
WHERE row._start IS NOT NULL AND row._end IS NOT NULL

MATCH (a {id: row._start})
MATCH (b {id: row._end})

MERGE (a)-[:CONNECTED_TO]->(b);


LOAD CSV WITH HEADERS FROM 'file:///paradise/intermediary_of.csv' AS row

WITH row
WHERE row._start IS NOT NULL AND row._end IS NOT NULL

MATCH (a {id: row._start})
MATCH (b {id: row._end})

MERGE (a)-[:INTERMEDIARY_OF]->(b);

LOAD CSV WITH HEADERS FROM 'file:///paradise/officer_of.csv' AS row

WITH row
WHERE row._start IS NOT NULL AND row._end IS NOT NULL

MATCH (a {id: row._start})
MATCH (b {id: row._end})

MERGE (a)-[:OFFICER_OF]->(b);

LOAD CSV WITH HEADERS FROM 'file:///paradise/registered_address.csv' AS row

WITH row
WHERE row._start IS NOT NULL AND row._end IS NOT NULL

MATCH (a {id: row._start})
MATCH (b {id: row._end})

MERGE (a)-[:REGISTERED_ADDRESS]->(b);

LOAD CSV WITH HEADERS FROM 'file:///paradise/same_id_as.csv' AS row

WITH row
WHERE row._start IS NOT NULL AND row._end IS NOT NULL

MATCH (a {id: row._start})
MATCH (b {id: row._end})

MERGE (a)-[:SAME_ID_AS]->(b);

LOAD CSV WITH HEADERS FROM 'file:///paradise/same_name_as.csv' AS row

WITH row
WHERE row._start IS NOT NULL AND row._end IS NOT NULL

MATCH (a {id: row._start})
MATCH (b {id: row._end})

MERGE (a)-[:SAME_NAME_AS]->(b);

match (a)-[r]->(b) set r.toDelete=False;


MATCH (e)-[:HAS_JURISDICTION]->(:Jurisdiction)<-[:HAS_JURISDICTION]-(e1)
WHERE e.incorporation_date = e1.incorporation_date
  AND e.name <> e1.name
  AND e <> e1
WITH e, e1
LIMIT 3000
SET e1.previous_name = e1.name,
    e1.name = e.name
RETURN e1.previous_name, e1.name;












WITH {
  Jan: '01', Feb: '02', Mar: '03', Apr: '04',
  May: '05', Jun: '06', Jul: '07', Aug: '08',
  Sep: '09', Oct: '10', Nov: '11', Dec: '12'
} AS months

MATCH (d:Date)
WITH d, split(d.value, '-') AS parts, months
WHERE size(parts) = 3 AND months[parts[1]] IS NOT NULL

SET d.value = date(
  parts[0] + '-' + months[parts[1]] + '-' + parts[2]
);

WITH date("1993-04-27") AS minDate,
     date("2029-12-18") AS maxDate,
     duration.inDays(date("1993-04-27"), date("2029-12-18")).days AS totalDays

MATCH ()-[r:OFFICER_OF]->()

WITH r, minDate, totalDays,
     toInteger(rand() * totalDays) AS startOffset,
     toInteger(rand() * totalDays) AS endOffset

WITH r,
     minDate + duration({days: startOffset}) AS d1,
     minDate + duration({days: endOffset}) AS d2

// ensure start <= end
SET r.start_date = CASE WHEN d1 <= d2 THEN d1 ELSE d2 END,
    r.end_date   = CASE WHEN d1 <= d2 THEN d2 ELSE d1 END;


CALL apoc.periodic.iterate(
"
MATCH (o)-[r:OFFICER_OF]->(e:Entity)-[relDate:HAS_CLOSED_DATE]->(d:Date)
WHERE r.end_date > d.value
RETURN DISTINCT o,e,d,r,relDate
",
"
WITH o,e,d,r,relDate,
     apoc.create.uuid() AS uuid,
     rand() AS randomValue

SET o.isViolation = true,
    e.isViolation = true,
    d.isViolation = true,
    r.isViolation = true,
    relDate.isViolation = true,

    o.violationId = coalesce(o.violationId, []) + uuid,
    e.violationId = coalesce(e.violationId, []) + uuid,
    d.violationId = coalesce(d.violationId, []) + uuid,
    r.violationId = coalesce(r.violationId, []) + uuid,
    relDate.violationId = coalesce(relDate.violationId, []) + uuid,
    r.violationType=1,
    relDate.violationType=1,
    r.end_date_GT = CASE
      WHEN randomValue < 0.5 THEN r.end_date
      ELSE d.value
    END,

    r.toDelete_GT = CASE
      WHEN randomValue < 0.5 THEN true
      ELSE false
    END,

    r.toDelete = CASE
      WHEN randomValue < 0.5 THEN true
      ELSE false
    END

CREATE (:Violation {
  violationId: uuid,
  violationType:1,
  repairs:
  '[MATCH ()-[r:OFFICER_OF]->() WHERE elementId(r)=$r SET r.deleted=true,
    MATCH ()-[r:OFFICER_OF]->() WHERE elementId(r)=$r SET r.end_date=$newEndDate,
    MATCH ()-[r:OFFICER_OF]->() WHERE elementId(r)=$r SET r.deleted=true, r.end_date=$newEndDate]',

  order: CASE
    WHEN randomValue < 0.5 THEN [0,2,1]
    ELSE [1,2,0]
  END,

  repair_GT: CASE
    WHEN randomValue < 0.5 THEN 0
    ELSE 1
  END,

  newEndDate_GT: d.value
})
",
{
  batchSize: 100,
  parallel: false
});



CALL apoc.periodic.iterate(
"
MATCH (e:Entity)-[r1:HAS_JURISDICTION]->(j:Jurisdiction)<-[r2:HAS_JURISDICTION]-(e1:Entity)
WHERE e <> e1
  AND e.incorporation_date = e1.incorporation_date
  AND e.name = e1.name
RETURN DISTINCT e,e1,j,r1,r2
",
"
WITH e,e1,j,r1,r2,
     apoc.create.uuid() AS uuid,
     rand() AS randomValue,
     e1.name + '_GT_FIX' AS newName

SET e.isViolation = true,
    e1.isViolation = true,
    j.isViolation = true,
    r1.isViolation = true,
    r2.isViolation = true,

    e.violationId = coalesce(e.violationId, []) + uuid,
    e1.violationId = coalesce(e1.violationId, []) + uuid,
    j.violationId = coalesce(j.violationId, []) + uuid,
    r1.violationId = coalesce(r1.violationId, []) + uuid,
    r2.violationId = coalesce(r2.violationId, []) + uuid,
    r1.violationType=3,
    r2.violationType=3,
    e1.name_GT = CASE
      WHEN randomValue < 0.5 THEN newName
      ELSE e1.name
    END,

    r2.toDelete_GT = CASE
      WHEN randomValue >= 0.5 THEN true
      ELSE false
    END,

    r2.toDelete = CASE
      WHEN randomValue >= 0.5 THEN true
      ELSE false
    END

CREATE (:Violation {
  violationId: uuid,
  violationType:3,
  repairs:
  '[MATCH (e1) WHERE elementId(e1)=$e1 SET e1.name=$newName,
    MATCH ()-[r2:HAS_JURISDICTION]->(:Jurisdiction) WHERE elementId(r2)=$r2 SET r2.deleted=true,
    MATCH (e1)-[r2:HAS_JURISDICTION]->(:Jurisdiction) WHERE elementId(e1)=$e1 AND elementId(r2)=$r2 SET e1.name=$newName, r2.deleted=true]',

  order: CASE
    WHEN randomValue < 0.5 THEN [0,2,1]
    ELSE [1,2,0]
  END,

  repair_GT: CASE
    WHEN randomValue < 0.5 THEN 0
    ELSE 1
  END,

  newName_GT: newName
})
",
{
  batchSize: 100,
  parallel: false
});



CALL apoc.periodic.iterate(
"
MATCH (c1:Country)-[rr]-(n:Entity)-[r:REGISTERED_ADDRESS]-(a:Address)-[l:LOCATED_IN]-(c:Country)
WHERE c1.code <> c.code
RETURN DISTINCT c1,n,a,c,rr,r,l
",
"
WITH c1,n,a,c,rr,r,l,
     apoc.create.uuid() AS uuid,
     rand() AS randomValue,
     c.code AS newCountryCode

SET c1.isViolation = true,
    n.isViolation = true,
    a.isViolation = true,
    c.isViolation = true,
    rr.isViolation = true,
    r.isViolation = true,
    l.isViolation = true,

    c1.violationId = coalesce(c1.violationId, []) + uuid,
    n.violationId = coalesce(n.violationId, []) + uuid,
    a.violationId = coalesce(a.violationId, []) + uuid,
    c.violationId = coalesce(c.violationId, []) + uuid,
    rr.violationId = coalesce(rr.violationId, []) + uuid,
    r.violationId = coalesce(r.violationId, []) + uuid,
    l.violationId = coalesce(l.violationId, []) + uuid,
    rr.violationType=2,
    r.violationType=2,
    l.violationType=2,

    c1.code_GT = CASE
      WHEN randomValue >= 0.5 THEN newCountryCode
      ELSE c1.code
    END,

    r.toDelete_GT = CASE
      WHEN randomValue < 0.5 THEN true
      ELSE false
    END,

    r.toDelete = CASE
      WHEN randomValue < 0.5 THEN true
      ELSE false
    END

CREATE (:Violation {
  violationId: uuid,
  violationType:2,
  repairs:
  '[MATCH (:Entity)-[r:REGISTERED_ADDRESS]-(:Address) WHERE elementId(r)=$r SET r.deleted=true,
    MATCH (c1:Country) WHERE elementId(c1)=$c1 SET c1.code=$newCountryCode,
    MATCH (c1:Country), (:Entity)-[r:REGISTERED_ADDRESS]-(:Address) WHERE elementId(c1)=$c1 AND elementId(r)=$r SET r.deleted=true, c1.code=$newCountryCode]',

  order: CASE
    WHEN randomValue < 0.5 THEN [0,2,1]
    ELSE [1,2,0]
  END,

  repair_GT: CASE
    WHEN randomValue < 0.5 THEN 0
    ELSE 1
  END,

  newCountryCode_GT: newCountryCode
})
",
{
  batchSize: 100,
  parallel: false
});