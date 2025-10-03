CREATE INDEX IF NOT EXISTS FOR (a:Person) ON (a.PersonId);
CREATE INDEX IF NOT EXISTS FOR (l:Tag) ON (l.TagId);
CREATE INDEX IF NOT EXISTS FOR (c:TagClass) ON (c.TagClassId);
CREATE INDEX IF NOT EXISTS FOR (p:Post) ON (p.PostId);
CREATE INDEX IF NOT EXISTS FOR (m:Forum) ON (m.ForumId);
CREATE INDEX IF NOT EXISTS FOR (p:Comment) ON (p.CommentId);

CALL apoc.periodic.iterate(
  "LOAD CSV WITH HEADERS FROM 'file:///snb/Person.csv' AS row FIELDTERMINATOR '|'RETURN row",
  "
CREATE (a:Person {
  PersonId: toInteger(row.id),
  creationDate: datetime(replace(row.creationDate, ' ', 'T')),
  firstName: toString(row.firstName),
  lastName: toString(row.lastName),
  gender: toString(row.gender),
  birthday: date(row.birthday),
  locationIP: toString(row.locationIP),
  browserUsed: toString(row.browserUsed),
  LocationCityId: toInteger(row.LocationCityId),
  language: toString(row.language),
  email: toString(row.emaill)
})",
  {batchSize: 1000, parallel: false}
);


CALL apoc.periodic.iterate(
  "LOAD CSV WITH HEADERS FROM 'file:///snb/TagClass.csv' AS row FIELDTERMINATOR '|'RETURN row",
  "
CREATE (tc:TagClass {
  TagClassId: toInteger(row.id),
  name: toString(row.name),
  url: toString(row.url)
})",
  {batchSize: 1000, parallel: false}
);



CALL apoc.periodic.iterate(
  "LOAD CSV WITH HEADERS FROM 'file:///snb/Tag.csv' as row FIELDTERMINATOR '|'RETURN row",
  "
CREATE (t:Tag {
  TagId: toInteger(row.id),
  name: toString(row.name),
  url: toString(row.url),
  TypeTagClassId: toInteger(row.TypeTagClassId)
})",
  {batchSize: 1000, parallel: false}
);



CALL apoc.periodic.iterate(
  "LOAD CSV WITH HEADERS FROM 'file:///snb/Post.csv' AS row FIELDTERMINATOR '|'RETURN row",
  "
CREATE (p:Post {
  PostId: toInteger(row.id),
  creationDate: datetime(replace(row.creationDate, ' ', 'T')),
  imageFile: toString(row.imageFile),
  locationIP: toString(row.locationIP),
  browserUsed: toString(row.browserUsed),
  language: toString(row.language),
  content: toString(row.content),
  length: toInteger(row.length)
})",
  {batchSize: 1000, parallel: false}
);


CALL apoc.periodic.iterate(
  "LOAD CSV WITH HEADERS FROM 'file:///snb/Forum.csv' AS row FIELDTERMINATOR '|'RETURN row",
  "
CREATE (f:Forum {
  ForumId: toInteger(row.id),
  creationDate: datetime(replace(row.creationDate, ' ', 'T')),
  title: toString(row.title)
})",
  {batchSize: 1000, parallel: false}
);


CALL apoc.periodic.iterate(
  "LOAD CSV WITH HEADERS FROM 'file:///snb/Comment.csv' AS row FIELDTERMINATOR '|'RETURN row",
  "
CREATE (c:Comment {
  CommentId: toInteger(row.id),
  creationDate: datetime(replace(row.creationDate, ' ', 'T')),
  locationIP: toString(row.locationIP),
  browserUsed: toString(row.browserUsed),
  content: toString(row.content),
  length: toInteger(row.length)
})",
  {batchSize: 1000, parallel: false}
);



CALL apoc.periodic.iterate(
  "LOAD CSV WITH HEADERS FROM 'file:///snb/Organisation.csv' AS row FIELDTERMINATOR '|'RETURN row",
  "
CREATE (c:Organisation {
  OrganisationId: toInteger(row.id),
  creationDate: datetime(replace(row.creationDate, ' ', 'T')),
  type: toString(row.type),
  name: toString(row.name),
  url: toString(row.url),
  LocationPlaceId: toInteger(row.LocationPlaceId)
})",
  {batchSize: 1000, parallel: false}
);

CALL apoc.periodic.iterate(
  "LOAD CSV WITH HEADERS FROM 'file:///snb/Place.csv' AS row FIELDTERMINATOR '|'RETURN row",
  "
CREATE (c:Place {
  PlaceId: toInteger(row.id),
  creationDate: datetime(replace(row.creationDate, ' ', 'T')),
  type: toString(row.type),
  name: toString(row.name),
  url: toString(row.url),
  PartOfPlaceId: toInteger(row.PartOfPlaceId)
})",
  {batchSize: 1000, parallel: false}
);


CALL apoc.periodic.iterate(
  "LOAD CSV WITH HEADERS FROM 'file:///snb/Comment_hasTag_Tag.csv' AS row FIELDTERMINATOR '|'RETURN row",
  "
MATCH (c:Comment {CommentId: toInteger(row.CommentId)}), (t:Tag {TagId: toInteger(row.TagId)})
CREATE (c)-[:HAS_TAG {
  creationDate: datetime(replace(row.creationDate, ' ', 'T'))
}]->(t)",
  {batchSize: 1000, parallel: false}
);


CALL apoc.periodic.iterate(
  "LOAD CSV WITH HEADERS FROM 'file:///snb/Forum_hasMember_Person.csv' AS row FIELDTERMINATOR '|'RETURN row",
  "
MATCH (f:Forum {ForumId: toInteger(row.ForumId)}), (p:Person {PersonId: toInteger(row.PersonId)})
CREATE (f)-[:HAS_MEMBER {
  creationDate: datetime(replace(row.creationDate, ' ', 'T'))
}]->(p)",
  {batchSize: 1000, parallel: false}
);

CALL apoc.periodic.iterate(
  "LOAD CSV WITH HEADERS FROM 'file:///snb/Person_hasInterest_Tag.csv' AS row FIELDTERMINATOR '|'RETURN row",
  "
MATCH (p:Person {PersonId: toInteger(row.PersonId)}), (t:Tag {TagId: toInteger(row.TagId)})
CREATE (p)-[:HAS_INTEREST {
  creationDate: datetime(replace(row.creationDate, ' ', 'T'))
}]->(t)",
  {batchSize: 1000, parallel: false}
);


CALL apoc.periodic.iterate(
  "LOAD CSV WITH HEADERS FROM 'file:///snb/Post_hasTag_Tag.csv' AS row FIELDTERMINATOR '|'RETURN row",
  "
MATCH (p:Post {PostId: toInteger(row.PostId)}), (t:Tag {TagId: toInteger(row.TagId)})
CREATE (p)-[:HAS_TAG {
  creationDate: datetime(replace(row.creationDate, ' ', 'T'))
}]->(t)",
  {batchSize: 1000, parallel: false}
);

CALL apoc.periodic.iterate(
  "LOAD CSV WITH HEADERS FROM 'file:///snb/Person_workAt_Company.csv' AS row FIELDTERMINATOR '|'RETURN row",
  "
MATCH (p:Person {PersonId: toInteger(row.PersonId)}), (c:Organisation {OrganisationId: toInteger(row.CompanyId)})
CREATE (p)-[:WORK_AT {
  creationDate: datetime(replace(row.creationDate, ' ', 'T')),
  workFrom: toInteger(row.workFrom)
}]->(c)",
  {batchSize: 1000, parallel: false}
);

CALL apoc.periodic.iterate(
  "LOAD CSV WITH HEADERS FROM 'file:///snb/Person_studyAt_University.csv' AS row FIELDTERMINATOR '|'RETURN row",
  "
MATCH (p:Person {PersonId: toInteger(row.PersonId)}), (u:Organisation {OrganisationId: toInteger(row.UniversityId)})
CREATE (p)-[:STUDY_AT {
  creationDate: datetime(replace(row.creationDate, ' ', 'T')),
  classYear: toInteger(row.classYear)
}]->(u)",
  {batchSize: 1000, parallel: false}
);

CALL apoc.periodic.iterate(
  "LOAD CSV WITH HEADERS FROM 'file:///snb/Person_likes_Post.csv' AS row FIELDTERMINATOR '|'RETURN row",
  "
MATCH (p:Person {PersonId: toInteger(row.PersonId)}), (post:Post {PostId: toInteger(row.PostId)})
CREATE (p)-[:LIKES {
  creationDate: datetime(replace(row.creationDate, ' ', 'T'))
}]->(post)",
  {batchSize: 1000, parallel: false}
);

CALL apoc.periodic.iterate(
  "LOAD CSV WITH HEADERS FROM 'file:///snb/Person_likes_Comment.csv' AS row FIELDTERMINATOR '|'RETURN row",
  "
MATCH (p:Person {PersonId: toInteger(row.PersonId)}), (c:Comment {CommentId: toInteger(row.CommentId)})
CREATE (p)-[:LIKES {
  creationDate: datetime(replace(row.creationDate, ' ', 'T'))
}]->(c)",
  {batchSize: 1000, parallel: false}
);

CALL apoc.periodic.iterate(
  "LOAD CSV WITH HEADERS FROM 'file:///snb/Person_knows_Person.csv' AS row FIELDTERMINATOR '|'RETURN row",
  "
MATCH (p1:Person {PersonId: toInteger(row.Person1Id)}), (p2:Person {PersonId: toInteger(row.Person2Id)})
CREATE (p1)-[:KNOWS {
  creationDate: datetime(replace(row.creationDate, ' ', 'T'))
}]->(p2)",
  {batchSize: 1000, parallel: false}
);

CALL apoc.periodic.iterate(
  "LOAD CSV WITH HEADERS FROM 'file:///snb/Person_hasInterest_Tag.csv' AS row FIELDTERMINATOR '|'RETURN row",
  "
MATCH (p:Person {PersonId: toInteger(row.PersonId)}), (t:Tag {TagId: toInteger(row.TagId)})
CREATE (p)-[:HAS_INTEREST {
  creationDate: datetime(replace(row.creationDate, ' ', 'T'))
}]->(t)",
  {batchSize: 1000, parallel: false}
);



CALL apoc.periodic.iterate(
  "LOAD CSV WITH HEADERS FROM 'file:///snb/Post.csv' AS row FIELDTERMINATOR '|'RETURN row",
  "
MATCH (o:Post {PostId: toInteger(row.id)}), (f:Forum {ForumId: toInteger(row.ContainerForumId)}), (pe:Person {PersonId: toInteger(row.CreatorPersonId)})
CREATE (o)-[:POSTED_IN {
  creationDate: datetime(replace(row.creationDate, ' ', 'T'))
}]->(f)
CREATE (pe)-[:CREATED {
  creationDate: datetime(replace(row.creationDate, ' ', 'T'))
}]->(o)
",
  {batchSize: 1000, parallel: false}
);



CALL apoc.periodic.iterate(
  "LOAD CSV WITH HEADERS FROM 'file:///snb/Comment.csv' AS row FIELDTERMINATOR '|'RETURN row",
  "
MATCH (o:Comment {CommentId: toInteger(row.id)}), (p:Post {PostId: toInteger(row.ParentPostId)}), (pe:Person {PersonId: toInteger(row.CreatorPersonId)})
CREATE (pe)-[:POSTED {
  creationDate: datetime(replace(row.creationDate, ' ', 'T'))
}]->(o)
CREATE (o)-[:TO {
  creationDate: datetime(replace(row.creationDate, ' ', 'T'))
}]->(p)
",
  {batchSize: 1000, parallel: false}
);


CALL apoc.periodic.iterate(
  "LOAD CSV WITH HEADERS FROM 'file:///snb/Comment.csv' AS row FIELDTERMINATOR '|'RETURN row",
  "
MATCH (o:Comment {CommentId: toInteger(row.id)}), (c:Comment {CommentId: toInteger(row.ParentCommentId)})
CREATE (o)-[:TO {
  creationDate: datetime(replace(row.creationDate, ' ', 'T'))
}]->(c)
",
  {batchSize: 1000, parallel: false}
);
CALL apoc.periodic.iterate(
  "LOAD CSV WITH HEADERS FROM 'file:///snb/Organisation.csv' AS row FIELDTERMINATOR '|'RETURN row",
  "
MATCH (o:Organisation {OrganisationId: toInteger(row.id)}), (p:Place {PlaceId: toInteger(row.LocationPlaceId)})
CREATE (o)-[:LOCATED_IN {
  creationDate: datetime(replace(row.creationDate, ' ', 'T'))
}]->(p)",
  {batchSize: 1000, parallel: false}
);



CALL apoc.periodic.iterate(
  "LOAD CSV WITH HEADERS FROM 'file:///snb/Person.csv' AS row FIELDTERMINATOR '|'RETURN row",
  "
MATCH (pe:Person {PersonId: toInteger(row.id)}), (p:Place {PlaceId: toInteger(row.LocationCityId)})
CREATE (pe)-[:LIVES_IN {
  creationDate: datetime(replace(row.creationDate, ' ', 'T'))
}]->(p)",
  {batchSize: 1000, parallel: false}
);

match c=(po:Post)<-[:CREATED]-(p:Person) with po, p limit 25000 merge (p)-[:LIKES]-(po);
match ()-[e]->() set e.toDelete = False;

MATCH (n)
WHERE n.creationDate is not null
WITH n, datetime(n.creationDate) AS dt
SET n.creationMillis = duration.between(datetime("1970-01-01T00:00:00Z"), dt).seconds * 1000 +
                       duration.between(datetime("1970-01-01T00:00:00Z"), dt).nanoseconds / 1000000;



//36800
CALL apoc.periodic.iterate(
  "
  MATCH (c:Comment)-[r:TO]->(c1:Post)
  WHERE c.creationMillis < c1.creationMillis
  RETURN DISTINCT c, c1, r
  ",
  "
  WITH c, c1, r, apoc.create.uuid() AS uuid, rand() * 10000 AS offset, rand() AS randomValue

  SET c.isViolation = true,
      c1.isViolation = true,
      r.isViolation = true,

      c.violationId = coalesce(c.violationId, []) + uuid,
      c1.violationId = coalesce(c1.violationId, []) + uuid,
      r.violationId = coalesce(r.violationId, []) + uuid

  FOREACH (_ IN CASE WHEN randomValue < 0.5 THEN [1] ELSE [] END |
    SET c.creationMillis_GT = c1.creationMillis + toInteger(offset)
  )
  FOREACH (_ IN CASE WHEN randomValue >= 0.5 THEN [1] ELSE [] END |
    SET c1.creationMillis_GT = c.creationMillis - toInteger(offset)
  )
  ",
  {
    batchSize: 100,
    parallel: false
  }
);

MATCH (p:Person)
WHERE p.birthday is not null
WITH p, date(p.birthday) AS birthdate, date() AS today
SET p.age = duration.between(birthdate, today).years;


MATCH (p:Person)
WITH p ORDER BY rand() LIMIT 500
SET p.age = toInteger(16 + rand() * (45 - 16 + 1));

MATCH (f:Forum)
SET f.ageRequirement = toInteger(16 + rand() * (21 - 16 + 1));

//4400
CALL apoc.periodic.iterate(
  "
  MATCH (p:Person)<-[m:HAS_MEMBER]-(f:Forum)
  WHERE p.age < f.ageRequirement
  RETURN DISTINCT p, f, m
  ",
  "
  WITH p, f, m, apoc.create.uuid() AS uuid, rand() AS randomValue, rand() * 5 AS offset

  SET p.isViolation = true,
      f.isViolation = true,
      m.isViolation = true,

      p.violationId = coalesce(p.violationId, []) + uuid,
      f.violationId = coalesce(f.violationId, []) + uuid,
      m.violationId = coalesce(m.violationId, []) + uuid

  FOREACH (_ IN CASE WHEN randomValue < 0.5 THEN [1] ELSE [] END |
    SET f.ageRequirement_GT = 16
  )
  FOREACH (_ IN CASE WHEN randomValue >= 0.5 THEN [1] ELSE [] END |
    SET p.age_GT = f.ageRequirement + toInteger(offset)
  )
  ",
  {
    batchSize: 100,
    parallel: false
  }
);


//3500 
CALL apoc.periodic.iterate(
  "
  MATCH (pl1:Place)<-[l:LIVES_IN]-(p:Person)-[:WORK_AT]->(o:Organisation)-[ln:LOCATED_IN]->(pl:Place)
  WHERE id(pl) <> id(pl1)
  RETURN DISTINCT p, o, pl, pl1, l, ln
  ",
  "
  WITH p, o, pl, pl1, l, ln, apoc.create.uuid() AS uuid, rand() AS randomValue

  SET p.isViolation = true,
      o.isViolation = true,
      pl.isViolation = true,
      pl1.isViolation = true,
      l.isViolation = true,
      ln.isViolation = true,

      p.violationId = coalesce(p.violationId, []) + uuid,
      o.violationId = coalesce(o.violationId, []) + uuid,
      pl.violationId = coalesce(pl.violationId, []) + uuid,
      pl1.violationId = coalesce(pl1.violationId, []) + uuid,
      l.violationId = coalesce(l.violationId, []) + uuid,
      ln.violationId = coalesce(ln.violationId, []) + uuid

  FOREACH (_ IN CASE WHEN randomValue < 0.5 THEN [1] ELSE [] END |
    SET l.toDelete_GT = true
  )
  FOREACH (_ IN CASE WHEN randomValue >= 0.5 THEN [1] ELSE [] END |
    SET ln.toDelete_GT = true
  )
  ",
  {
    batchSize: 100,
    parallel: false
  }
);


CALL apoc.periodic.iterate(
  "
  MATCH (po:Post)<-[c:CREATED]-(p:Person)-[l:LIKES]-(po)
  RETURN DISTINCT p, po, c, l
  ",
  "
  WITH p, po, c, l, apoc.create.uuid() AS uuid, rand() AS randomValue

  SET p.isViolation = true,
      po.isViolation = true,
      c.isViolation = true,
      l.isViolation = true,

      p.violationId = coalesce(p.violationId, []) + uuid,
      po.violationId = coalesce(po.violationId, []) + uuid,
      c.violationId = coalesce(c.violationId, []) + uuid,
      l.violationId = coalesce(l.violationId, []) + uuid

  FOREACH (_ IN CASE WHEN randomValue < 0.5 THEN [1] ELSE [] END |
    SET c.toDelete_GT = true
  )
  FOREACH (_ IN CASE WHEN randomValue >= 0.5 THEN [1] ELSE [] END |
    SET l.toDelete_GT = true
  )
  ",
  {
    batchSize: 100,
    parallel: false
  }
);





